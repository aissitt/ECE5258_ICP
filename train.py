import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import gc
from tensorflow.keras import backend as K
from datetime import datetime
import argparse
import json
import time
from model import unet_model
from loss import data_driven_loss, physics_informed_loss
from metrics import RMSEPerComponent, NRMSEPerComponent, MAEPerComponent, NMAEPerComponent
from utils import load_and_split_data, plot_training_history, create_output_directories, save_hyperparameters, log_runtime
from keras.saving import register_keras_serializable

# Enable dynamic memory allocation for GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Argument parser for command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--output-dir', required=True, help='Base directory to save training results')
parser.add_argument('--mode', choices=['data', 'physics'], required=True, help='Training mode: data-driven or physics-informed')
parser.add_argument('--config', help='Path to custom config file', default='config.json')
parser.add_argument('--use-tuning', action='store_true', help='Flag to use the best configuration from tuning')
parser.add_argument('--config-path', help='Path to specific config file (overrides default or tuning path)')
parser.add_argument('--seed', type=int, help='Set a seed for reproducibility (overrides config or tuning seed)')

args = parser.parse_args()

# Load configuration
if args.use_tuning:
    # Base directory for tuning outputs
    tuning_dir = os.path.join("/home1/aissitt2019/Hemodynamics/LVAD/tuning_outputs", args.mode)
    
    # Locate the most recent tuning run directory
    if os.path.exists(tuning_dir) and os.listdir(tuning_dir):
        tuning_run_dir = max([os.path.join(tuning_dir, d) for d in os.listdir(tuning_dir)], key=os.path.getmtime)
    else:
        raise FileNotFoundError(f"No tuning runs found in {tuning_dir}. Please ensure you have performed a tuning run first.")

    # Locate the most recent config file within the tuning run
    best_config_path = os.path.join(tuning_run_dir, 'logs', f'best_config_{args.mode}.json')
    if os.path.exists(best_config_path):
        with open(best_config_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded best configuration from {best_config_path}")

        # Load the seed from the best config if present
        config_seed = config["training"].get("seed", None)
    else:
        raise FileNotFoundError(f"Best configuration not found in {tuning_run_dir}.")
elif args.config_path:
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    print(f"Loaded configuration from {args.config_path}")

    # Check if a seed is provided in the config file
    config_seed = config["training"].get("seed", None)
else:
    # Load default or specified config
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        with open('config.json', 'r') as f:
            config = json.load(f)

# Add mode from args to config
config["mode"] = args.mode  # This ensures "mode" is available in the config dictionary.

# Resolve environment variables for input and output data paths
config["training"]["input_data"] = os.getenv("INPUT_DATA_PATH", config["training"]["input_data"])
config["training"]["output_data"] = os.getenv("OUTPUT_DATA_PATH", config["training"]["output_data"])

# Set the seed for reproducibility (override config/tuning seed if specified)
config_seed = config["training"].get("seed", None)
seed = args.seed if args.seed else config_seed

if seed:
    print(f"Setting seed: {seed}")
    np.random.seed(seed)
    tf.random.set_seed(seed)
else:
    print("No seed specified. Results may vary.")

# Create directories for output
output_dir = args.output_dir
logs_dir = os.path.join(output_dir, "logs")
images_dir = os.path.join(output_dir, "images")
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(images_dir, exist_ok=True)

# Load and split data using indices provided in config
trainX_np, trainY_np, valX_np, valY_np = load_and_split_data(
    config["training"]["input_data"],
    config["training"]["output_data"],
    config["training"]["train_indices"],
    config["training"]["val_indices"]
)

# Define named functions for loss
@register_keras_serializable()
def data_loss_fn(y_true, y_pred):
    # Extract RDF component from y_true to create a mask
    mask = y_true[..., 0] > 0
    return data_driven_loss(y_true, y_pred, config, mask)

@register_keras_serializable()
def physics_loss_fn(y_true, y_pred):
    # Extract RDF component from y_true to create a mask
    mask = y_true[..., 0] > 0  
    return physics_informed_loss(y_true, y_pred, config, mask)

# Define custom objects for saving and loading
custom_objects = {
    "data_loss_fn": data_loss_fn,
    "physics_loss_fn": physics_loss_fn,
    "RMSEPerComponent": RMSEPerComponent,
    "NRMSEPerComponent": NRMSEPerComponent,
    "MAEPerComponent": MAEPerComponent,
    "NMAEPerComponent": NMAEPerComponent
}

# Define the tuning-compatible train function
def train_tuning(config, arch_params=None, output_dir=None):
    K.clear_session()
    gc.collect()

    # Strategy for distributed training across GPUs
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        input_shape = tuple(config["model"]["input_shape"])
        model = unet_model(
            input_shape,
            activation=arch_params.get("activation", "relu"),
            batch_norm=False,
            dropout_rate=0.0,
            l2_reg=0.0,
            attention=False  # Turn off attention during tuning
        )

        # Compile the model
        if config["mode"] == 'data':
            model.compile(
                loss=data_loss_fn,
                optimizer=Adam(learning_rate=config["training"]["learning_rate"]),
                metrics=[RMSEPerComponent(), NRMSEPerComponent(), MAEPerComponent(), NMAEPerComponent()]
            )
        else:
            model.compile(
                loss=physics_loss_fn,
                optimizer=Adam(learning_rate=config["training"]["learning_rate"]),
                metrics=[RMSEPerComponent(), NRMSEPerComponent(), MAEPerComponent(), NMAEPerComponent()]
            )

        # Callbacks
        checkpoint_path = os.path.join(output_dir, f"lvad_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5")
        checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, verbose=1, monitor='val_loss', mode='min')
        early_stopping = EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss', mode='min')

        # Train the model
        history = model.fit(
            trainX_np, trainY_np,
            validation_data=(valX_np, valY_np),
            epochs=config["training"]["epochs"],
            batch_size=config["training"]["batch_size"] * strategy.num_replicas_in_sync,
            callbacks=[early_stopping, checkpoint]
        )

    K.clear_session()
    gc.collect()

    return float(min(history.history['val_loss']))

# Define the full-functionality train function
def train_full(config, arch_params=None, output_dir=None):
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        input_shape = tuple(config["model"]["input_shape"])
        model = unet_model(
            input_shape,
            activation=arch_params.get("activation", "relu") if arch_params else "relu",
            batch_norm=arch_params.get("batch_norm", False) if arch_params else False,
            dropout_rate=arch_params.get("dropout_rate", 0.0) if arch_params else 0.0,
            l2_reg=arch_params.get("l2_reg", 0.0) if arch_params else 0.0,
            attention=arch_params.get("attention", False) if arch_params else False,
        )

        if config["mode"] == 'data':
            model.compile(
                loss=data_loss_fn,
                optimizer=Adam(learning_rate=config["training"]["learning_rate"]),
                metrics=[RMSEPerComponent(), NRMSEPerComponent(), MAEPerComponent(), NMAEPerComponent()]
            )
        else:
            model.compile(
                loss=physics_loss_fn,
                optimizer=Adam(learning_rate=config["training"]["learning_rate"]),
                metrics=[RMSEPerComponent(), NRMSEPerComponent(), MAEPerComponent(), NMAEPerComponent()]
            )

        # Callbacks
        checkpoint_path = os.path.join(output_dir, f"lvad_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5")
        checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, verbose=1, monitor='val_loss', mode='min')
        early_stopping = EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss', mode='min')

        # Train the model
        history = model.fit(
            trainX_np, trainY_np,
            validation_data=(valX_np, valY_np),
            epochs=config["training"]["epochs"],
            batch_size=config["training"]["batch_size"] * strategy.num_replicas_in_sync,
            callbacks=[early_stopping, checkpoint]
        )

    return history, min(history.history['val_loss'])

# Main function to handle either tuning or full mode
if __name__ == '__main__':
    start_time = time.time()

    # Initialize arch_params as an empty dictionary if not using tuning
    arch_params = None

    if args.use_tuning:
        # Load the best architecture parameters from the tuning run
        arch_params = config.get("model", {})  # Set to an empty dictionary if not present
        print(f"Loaded architecture parameters: {arch_params}")

    # Run training using the appropriate function
    if args.use_tuning:
        val_loss = train_tuning(config, arch_params or {}, output_dir)  # Ensure arch_params is not None
    else:
        history, val_loss = train_full(config, arch_params or {}, output_dir)

    end_time = time.time()
    total_time = end_time - start_time

    # Plot training history and save metrics
    if not args.use_tuning:
        plot_training_history(history, images_dir)

    save_hyperparameters(config, output_dir)
    log_runtime(history.epoch[-1] + 1 if not args.use_tuning else 0, logs_dir)

    print(f"Training complete. Total time: {total_time:.2f} seconds. Final validation loss: {val_loss}")