import tensorflow as tf
import numpy as np

# Masking utility
def apply_mask(tensor, mask):
    expanded_mask = tf.expand_dims(mask, axis=-1)  # Expand last dimension of mask
    while len(expanded_mask.shape) < len(tensor.shape):
        expanded_mask = tf.expand_dims(expanded_mask, axis=-1)
    return tf.where(expanded_mask, tensor, tf.constant(np.nan, dtype=tensor.dtype))

# Debugging utility for gradients
def debug_gradients(grad, label, enable_debug=True):
    if enable_debug:
        grad_min = tf.reduce_min(grad)
        grad_max = tf.reduce_max(grad)
        grad_mean = tf.reduce_mean(grad)
        tf.print(f"[DEBUG] {label}: min={grad_min}, max={grad_max}, mean={grad_mean}")

# Centralized gradient calculation
def compute_spatial_gradients(y_pred, mask, enable_debug=False):
    y_pred_masked = apply_mask(y_pred, mask)

    # Use persistent GradientTape for multiple gradient computations
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(y_pred_masked)
        u, v, w = y_pred_masked[..., 0], y_pred_masked[..., 1], y_pred_masked[..., 2]

        # Compute the gradients
        du_dx = tape.gradient(u, y_pred_masked)[..., 0]
        du_dy = tape.gradient(u, y_pred_masked)[..., 1]
        du_dz = tape.gradient(u, y_pred_masked)[..., 2]
        dv_dx = tape.gradient(v, y_pred_masked)[..., 0]
        dv_dy = tape.gradient(v, y_pred_masked)[..., 1]
        dv_dz = tape.gradient(v, y_pred_masked)[..., 2]
        dw_dx = tape.gradient(w, y_pred_masked)[..., 0]
        dw_dy = tape.gradient(w, y_pred_masked)[..., 1]
        dw_dz = tape.gradient(w, y_pred_masked)[..., 2]

    del tape  # Explicitly delete the tape to free memory

    if enable_debug:
        debug_gradients(du_dx, "du/dx")
        debug_gradients(dv_dy, "dv/dy")
        debug_gradients(dw_dz, "dw/dz")
        debug_gradients(du_dy, "du/dy")
        debug_gradients(du_dz, "du/dz")
        debug_gradients(dv_dx, "dv/dx")
        debug_gradients(dv_dz, "dv/dz")
        debug_gradients(dw_dx, "dw/dx")
        debug_gradients(dw_dy, "dw/dy")

    return du_dx, dv_dy, dw_dz, du_dy, du_dz, dv_dx, dv_dz, dw_dx, dw_dy

# Compute vorticity
def compute_vorticity(y_pred, mask=None):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(y_pred)
        u, v, w = y_pred[..., 0], y_pred[..., 1], y_pred[..., 2]
        du_dy = tape.gradient(u, y_pred)[..., 1]
        du_dz = tape.gradient(u, y_pred)[..., 2]
        dv_dx = tape.gradient(v, y_pred)[..., 0]
        dv_dz = tape.gradient(v, y_pred)[..., 2]
        dw_dx = tape.gradient(w, y_pred)[..., 0]
        dw_dy = tape.gradient(w, y_pred)[..., 1]

    del tape

    omega_x = dw_dy - dv_dz
    omega_y = du_dz - dw_dx
    omega_z = dv_dx - du_dy

    vorticity = tf.stack([omega_x, omega_y, omega_z], axis=-1)

    if mask is not None:
        vorticity = apply_mask(vorticity, mask)

    return vorticity

# Data-Driven Loss
def data_driven_loss(y_true, y_pred, config, mask):
    delta = config['loss_parameters']['data_driven']['huber_delta']
    y_true = apply_mask(y_true, mask)
    y_pred = apply_mask(y_pred, mask)
    huber_loss = tf.keras.losses.Huber(delta=delta, reduction=tf.keras.losses.Reduction.SUM)
    return (huber_loss(y_true[..., 0], y_pred[..., 0]) +
            huber_loss(y_true[..., 1], y_pred[..., 1]) +
            huber_loss(y_true[..., 2], y_pred[..., 2])) / 3.0

# Continuity Loss
def compute_continuity_loss(grad_x, grad_y, grad_z):
    continuity_residual = grad_x + grad_y + grad_z
    return tf.reduce_mean(tf.square(continuity_residual))

# Momentum Loss
def compute_momentum_loss(du_dx, dv_dy, dw_dz, du_dy, du_dz, dv_dx, dv_dz, dw_dx, dw_dy, u, v, w, nu):
    # Convective terms
    conv_u = u * du_dx + v * du_dy + w * du_dz
    conv_v = u * dv_dx + v * dv_dy + w * dv_dz
    conv_w = u * dw_dx + v * dw_dy + w * dw_dz

    # Diffusion terms
    diffusion_u = nu * (du_dx + du_dy + du_dz)
    diffusion_v = nu * (dv_dx + dv_dy + dv_dz)
    diffusion_w = nu * (dw_dx + dw_dy + dw_dz)

    # Momentum residual
    momentum_residual = tf.stack(
        [conv_u - diffusion_u, conv_v - diffusion_v, conv_w - diffusion_w], axis=-1
    )
    return tf.reduce_mean(tf.square(momentum_residual))

# Gradient Penalty
def compute_gradient_penalty(grad_x, grad_y, grad_z):
    grad_magnitude = tf.sqrt(tf.square(grad_x) + tf.square(grad_y) + tf.square(grad_z))
    return tf.reduce_mean(grad_magnitude)

# Focused Vorticity Loss
def focused_vorticity_loss(y_true, y_pred, threshold, mask):
    y_true = apply_mask(y_true, mask)
    y_pred = apply_mask(y_pred, mask)

    # Compute vorticity for ground truth and predictions
    vorticity_true = compute_vorticity(y_true)
    vorticity_pred = compute_vorticity(y_pred)

    # Magnitude of vorticity
    vorticity_magnitude_true = tf.sqrt(tf.reduce_sum(tf.square(vorticity_true), axis=-1) + 1e-10)
    high_vorticity_mask = tf.expand_dims(vorticity_magnitude_true > threshold, axis=-1)

    # Compute vorticity error
    vorticity_error = tf.square(vorticity_true - vorticity_pred)
    focused_error = tf.where(high_vorticity_mask, vorticity_error, tf.zeros_like(vorticity_error))

    # Normalize by the high vorticity area
    high_vorticity_area = tf.reduce_sum(tf.cast(high_vorticity_mask, tf.float32))
    return tf.reduce_sum(focused_error) / (high_vorticity_area + 1e-10)

# Sobel 3D Loss
def sobel_3d_loss(y_true, y_pred, mask):
    y_true = apply_mask(y_true, mask)
    y_pred = apply_mask(y_pred, mask)
    sobel_kernel = tf.constant([
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    ], dtype=tf.float32)
    sobel_kernel = tf.reshape(sobel_kernel, [3, 3, 3, 1, 1])
    num_channels = y_true.shape[-1]
    sobel_kernel = tf.tile(sobel_kernel, [1, 1, 1, num_channels, 1])

    grad_true = tf.nn.conv3d(y_true, sobel_kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
    grad_pred = tf.nn.conv3d(y_pred, sobel_kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
    return tf.reduce_mean(tf.square(grad_true - grad_pred))

# Laplacian 3D Loss
def laplacian_3d_loss(y_pred, mask, epsilon=1e-6):
    y_pred = apply_mask(y_pred, mask)
    laplacian_kernel = tf.constant([
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        [[0, 1, 0], [1, -6, 1], [0, 1, 0]],
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    ], dtype=tf.float32)
    laplacian_kernel = tf.reshape(laplacian_kernel, [3, 3, 3, 1, 1])
    num_channels = y_pred.shape[-1]
    laplacian_kernel = tf.tile(laplacian_kernel, [1, 1, 1, num_channels, 1])
    
    # Compute Laplacian of y_pred
    laplacian = tf.nn.conv3d(y_pred, laplacian_kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
    
    # Penalize small Laplacian values to discourage smoothness
    penalty = 1.0 / (tf.square(laplacian) + epsilon)
    return tf.reduce_mean(penalty)

def anisotropy_loss(gradients, mask):
    du_dx, dv_dy, dw_dz, du_dy, du_dz, dv_dx, dv_dz, dw_dx, dw_dy = gradients

    # Stack gradients into a tensor for Jacobian computation
    gradients_stack = tf.stack([
        tf.stack([du_dx, dv_dx, dw_dx], axis=-1),
        tf.stack([du_dy, dv_dy, dw_dy], axis=-1),
        tf.stack([du_dz, dv_dz, dw_dz], axis=-1),
    ], axis=-2)  # Shape: [batch_size, depth, height, width, 3, 3]

    # Apply mask to gradients lazily (only on eigenvalues)
    batch_size, depth, height, width, _, _ = gradients_stack.shape
    gradients_reshaped = tf.reshape(gradients_stack, [-1, 3, 3])  # Shape: [batch_size * depth * height * width, 3, 3]

    # Compute eigenvalues
    eig_values = tf.linalg.eigvalsh(gradients_reshaped)  # Shape: [batch_size * depth * height * width, 3]

    # Compute anisotropy metric (e.g., max eigenvalue / min eigenvalue)
    max_eigen = tf.reduce_max(eig_values, axis=-1)
    min_eigen = tf.reduce_min(eig_values, axis=-1)
    anisotropy = max_eigen / (min_eigen + 1e-10)

    # Reshape anisotropy back to spatial dimensions
    anisotropy = tf.reshape(anisotropy, [batch_size, depth, height, width])  # Shape: [batch_size, depth, height, width]

    # Mask the final anisotropy values
    anisotropy_masked = tf.where(mask, anisotropy, tf.zeros_like(anisotropy))
    return tf.reduce_mean(anisotropy_masked)

# Full Physics-Informed Loss
def physics_informed_loss(y_true, y_pred, config, mask):
    hp = config['loss_parameters']['physics_informed']

    # Compute spatial gradients
    gradients = compute_spatial_gradients(y_pred, mask, enable_debug=False)
    du_dx, dv_dy, dw_dz, du_dy, du_dz, dv_dx, dv_dz, dw_dx, dw_dy = gradients
    u, v, w = y_pred[..., 0], y_pred[..., 1], y_pred[..., 2]

    # Compute individual loss terms
    data_loss = hp['lambda_data'] * data_driven_loss(y_true, y_pred, config, mask)
    continuity_loss = hp['lambda_continuity'] * compute_continuity_loss(du_dx, dv_dy, dw_dz)
    momentum_loss = hp['lambda_momentum'] * compute_momentum_loss(
        du_dx, dv_dy, dw_dz, du_dy, du_dz, dv_dx, dv_dz, dw_dx, dw_dy, u, v, w, hp['nu']
    )
    gradient_penalty = hp['lambda_gradient_penalty'] * compute_gradient_penalty(du_dx, dv_dy, dw_dz)
    vorticity_loss = hp['lambda_vorticity_focused'] * focused_vorticity_loss(
        y_true, y_pred, hp['threshold_vorticity'], mask
    )
    sobel_loss = hp['lambda_sobel'] * sobel_3d_loss(y_true, y_pred, mask)
    laplacian_loss = hp['lambda_laplacian'] * laplacian_3d_loss(y_pred, mask)
    anisotropy = hp['lambda_anisotropy'] * anisotropy_loss(gradients, mask)

    # Combine losses
    total_loss = (
        data_loss + continuity_loss + momentum_loss +
        gradient_penalty + vorticity_loss + sobel_loss +
        laplacian_loss + anisotropy
    )
    return total_loss