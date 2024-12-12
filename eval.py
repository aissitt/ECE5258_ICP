import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse
import json
from keras.utils import custom_object_scope
from datetime import datetime
from metrics import RMSEPerComponent, NRMSEPerComponent, MAEPerComponent, NMAEPerComponent
from loss import data_driven_loss, physics_informed_loss
from utils import load_dataset, apply_mask, create_output_directories, compute_errors, compute_high_error_metrics, plot_error_maps
import gc

# Define named functions for loss (same as in train.py)
def data_loss_fn(y_true, y_pred):
    return data_driven_loss(y_true, y_pred, config)

def physics_loss_fn(y_true, y_pred):
    return physics_informed_loss(y_true, y_pred, config)

# Define custom objects for loading the model
custom_objects = {
    "Custom>data_loss_fn": data_loss_fn,
    "Custom>physics_loss_fn": physics_loss_fn,
    "RMSEPerComponent": RMSEPerComponent,
    "NRMSEPerComponent": NRMSEPerComponent,
    "MAEPerComponent": MAEPerComponent,
    "NMAEPerComponent": NMAEPerComponent
}

# Argument parser for command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model-path', required=True, help='Path to the trained model file')
parser.add_argument('--mode', required=True, choices=['data', 'physics'], help='Type of the model')
parser.add_argument('--output-dir', required=True, help='Directory to save evaluation results')
parser.add_argument('--test-indices', nargs=2, type=int, help='Start and end indices for the test data')
parser.add_argument('--config', help='Path to custom config file', default='config.json')
args = parser.parse_args()

# Load configuration
if args.config:
    with open(args.config, 'r') as f:
        config = json.load(f)
else:
    with open('config.json', 'r') as f:
        config = json.load(f)

# If using environment variables, replace placeholders with their values
if config.get("use_env_vars", False):
    config["training"]["input_data"] = os.getenv("INPUT_DATA_PATH")
    config["training"]["output_data"] = os.getenv("OUTPUT_DATA_PATH")

# Determine test indices to use
if args.test_indices:
    test_indices = args.test_indices
else:
    test_indices = config["training"]["test_indices"]

# Load dataset using test indices from config or arguments
testX_np = load_dataset(config["training"]["input_data"])[test_indices[0]:test_indices[1]]
testY_np = load_dataset(config["training"]["output_data"])[test_indices[0]:test_indices[1]]

# Extract RDF component to create a mask
rdf = testX_np[..., 0]
mask = rdf > 0

# Clear previous session and garbage collect
tf.keras.backend.clear_session()
gc.collect()

# Load model with custom scope
with custom_object_scope(custom_objects):
    best_model = tf.keras.models.load_model(args.model_path)

# Evaluate the model on test data to obtain metrics
batch_size = config.get('evaluation', {}).get('batch_size', 1)  # Explicitly set batch size to control memory usage
results = best_model.evaluate(testX_np, testY_np, batch_size=batch_size, return_dict=True)
# results = best_model.evaluate(testX_np, testY_np, return_dict=True)
print(f"Evaluation results: {results}")

# Convert any NumPy arrays in results to lists for JSON serialization
for key in results:
    if isinstance(results[key], np.ndarray):
        results[key] = results[key].tolist()

# Clear previous session and garbage collect before prediction
tf.keras.backend.clear_session()
gc.collect()

# Compute additional error metrics
abs_errors, peak_error, peak_error_coords = compute_errors(testY_np, best_model.predict(testX_np), mask)
high_error_threshold = config["loss_parameters"].get("high_error_threshold", 0.01)  # Default to 0.01 if not specified
high_error_count, high_error_percentage = compute_high_error_metrics(abs_errors, high_error_threshold)

# Log results
metrics = {
    "Peak Error": float(peak_error),
    "Coordinates of Peak Error": [int(coord) for coord in peak_error_coords],
    "Number of high error points": int(high_error_count),
    "Percentage of high error points": float(high_error_percentage)
}

# Combine evaluation results with custom metrics
metrics.update(results)

# Create output directories based on the evaluation mode
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir, logs_dir, images_dir = create_output_directories(args.output_dir, '')

# Save metrics to a JSON file
metrics_file = os.path.join(logs_dir, "metrics.json")
with open(metrics_file, 'w') as f:
    json.dump(metrics, f, indent=4)
print(f"Metrics saved to {metrics_file}")

# Clear previous session and garbage collect before plotting
tf.keras.backend.clear_session()
gc.collect()

# Plot and save error maps
plot_error_maps(testY_np, best_model.predict(testX_np), abs_errors, peak_error_coords, images_dir, mask)

# Visualization setup for comparing true vs. predicted velocity components
y_slices = [50, 75, 85]
fig, axes = plt.subplots(3, len(y_slices) * 2, figsize=(23, 10), sharey=True)
fig.suptitle("Comparison of True vs. Predicted Velocity Components (Vx, Vy, Vz) Along Y-axis")

# Mask needs to be applied only for spatial dimensions, so reduce mask to 2D per slice
mask_2d = mask[0, :, :, :]  # Take the first example's mask and reduce to spatial dimensions

for i, y in enumerate(y_slices):
    for component in range(3):  # For Vx, Vy, Vz components
        true_slice = np.ma.masked_where(~mask_2d[:, y, :], testY_np[0, :, y, :, component])
        predicted_slice = np.ma.masked_where(~mask_2d[:, y, :], best_model.predict(testX_np)[0, :, y, :, component])

        # Compute vmin and vmax for each true/predicted pair
        vmin = min(true_slice.min(), predicted_slice.min())
        vmax = max(true_slice.max(), predicted_slice.max())

        # True slice visualization
        ax_true = axes[component, i * 2]
        im_true = ax_true.imshow(true_slice, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
        ax_true.set_title(f"True V{['x', 'y', 'z'][component]} Slice @ Y={y}")
        fig.colorbar(im_true, ax=ax_true)

        # Predicted slice visualization
        ax_pred = axes[component, i * 2 + 1]
        im_pred = ax_pred.imshow(predicted_slice, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
        ax_pred.set_title(f"Predicted V{['x', 'y', 'z'][component]} Slice @ Y={y}")
        fig.colorbar(im_pred, ax=ax_pred)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(images_dir, 'velocity_components_comparison_xz.png'))
plt.show()
