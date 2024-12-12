import tensorflow as tf

# RMSE per component as a custom metric
class RMSEPerComponent(tf.keras.metrics.Metric):
    def __init__(self, name="rmse_per_component", **kwargs):
        super().__init__(name=name, **kwargs)
        self.sum_squared_errors = self.add_weight(name="sse", initializer="zeros", shape=(3,), dtype=tf.float32)
        self.total_count = self.add_weight(name="count", initializer="zeros", dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Calculate squared errors and update state
        error_squared = tf.square(y_true - y_pred)
        error_sum = tf.reduce_sum(tf.cast(error_squared, tf.float32), axis=[0, 1, 2, 3])
        self.sum_squared_errors.assign_add(error_sum)
        self.total_count.assign_add(tf.cast(tf.size(y_true) / 3, tf.float32))  # 3 for the number of components

    def result(self):
        # Calculate RMSE
        mse = self.sum_squared_errors / self.total_count
        return tf.sqrt(mse)

    def reset_state(self):
        # Reset metric state
        self.sum_squared_errors.assign(tf.zeros(shape=(3,), dtype=tf.float32))
        self.total_count.assign(0.0)

# Normalized RMSE per component as a custom metric
class NRMSEPerComponent(RMSEPerComponent):
    def __init__(self, name="nrmse_per_component", **kwargs):
        super().__init__(name=name, **kwargs)
        self.min_values = self.add_weight(name="min_vals", shape=(3,), initializer=tf.keras.initializers.constant(value=float('inf')), dtype=tf.float32)
        self.max_values = self.add_weight(name="max_vals", shape=(3,), initializer=tf.keras.initializers.constant(value=float('-inf')), dtype=tf.float32)
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Update state with RMSE and min/max values
        super().update_state(y_true, y_pred, sample_weight)
        self.min_values.assign(tf.minimum(self.min_values, tf.cast(tf.reduce_min(y_true, axis=[0, 1, 2, 3]), tf.float32)))
        self.max_values.assign(tf.maximum(self.max_values, tf.cast(tf.reduce_max(y_true, axis=[0, 1, 2, 3]), tf.float32)))

    def result(self):
        # Calculate NRMSE
        rmse = super().result()
        range_values = self.max_values - self.min_values
        return rmse / (range_values + tf.keras.backend.epsilon())

# MAE per component as a custom metric
class MAEPerComponent(tf.keras.metrics.Metric):
    def __init__(self, name="mae_per_component", **kwargs):
        super().__init__(name=name, **kwargs)
        self.sum_absolute_errors = self.add_weight(name="sae", initializer="zeros", shape=(3,), dtype=tf.float32)
        self.total_count = self.add_weight(name="count", initializer="zeros", dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Calculate absolute errors and update state
        errors = tf.abs(y_true - y_pred)
        error_sum = tf.reduce_sum(tf.cast(errors, tf.float32), axis=[0, 1, 2, 3])
        self.sum_absolute_errors.assign_add(error_sum)
        self.total_count.assign_add(tf.cast(tf.size(y_true) / 3, tf.float32))  # 3 for the number of components

    def result(self):
        # Calculate MAE
        return self.sum_absolute_errors / self.total_count

    def reset_state(self):
        # Reset metric state
        self.sum_absolute_errors.assign(tf.zeros(shape=(3,), dtype=tf.float32))
        self.total_count.assign(0.0)

# Normalized MAE per component as a custom metric
class NMAEPerComponent(MAEPerComponent):
    def __init__(self, name="nmae_per_component", **kwargs):
        super().__init__(name=name, **kwargs)
        self.min_values = self.add_weight(name="min_vals", shape=(3,), initializer=tf.keras.initializers.constant(value=float('inf')), dtype=tf.float32)
        self.max_values = self.add_weight(name="max_vals", shape=(3,), initializer=tf.keras.initializers.constant(value=float('-inf')), dtype=tf.float32)
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Update state with MAE and min/max values
        super().update_state(y_true, y_pred, sample_weight)
        self.min_values.assign(tf.minimum(self.min_values, tf.cast(tf.reduce_min(y_true, axis=[0, 1, 2, 3]), tf.float32)))
        self.max_values.assign(tf.maximum(self.max_values, tf.cast(tf.reduce_max(y_true, axis=[0, 1, 2, 3]), tf.float32)))

    def result(self):
        # Calculate NMAE
        mae = super().result()
        range_values = self.max_values - self.min_values
        return mae / (range_values + tf.keras.backend.epsilon())

# Manual calculation for RMSE
def manual_rmse(y_true, y_pred):
    # Compute Root Mean Squared Error manually
    mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=[0, 1, 2, 3])
    rmse = tf.sqrt(mse)
    return rmse

# Manual calculation for NRMSE
def manual_nrmse(y_true, y_pred):
    # Compute Normalized Root Mean Squared Error manually
    rmse = manual_rmse(y_true, y_pred)
    min_val = tf.reduce_min(y_true, axis=[0, 1, 2, 3])
    max_val = tf.reduce_max(y_true, axis=[0, 1, 2, 3])
    nrmse = rmse / (max_val - min_val + tf.keras.backend.epsilon())
    return nrmse

# Manual calculation for MAE
def manual_mae(y_true, y_pred):
    # Compute Mean Absolute Error manually
    mae = tf.reduce_mean(tf.abs(y_true - y_pred), axis=[0, 1, 2, 3])
    return mae

# Manual calculation for NMAE
def manual_nmae(y_true, y_pred):
    # Compute Normalized Mean Absolute Error manually
    mae = manual_mae(y_true, y_pred)
    min_val = tf.reduce_min(y_true, axis=[0, 1, 2, 3])
    max_val = tf.reduce_max(y_true, axis=[0, 1, 2, 3])
    nmae = mae / (max_val - min_val + tf.keras.backend.epsilon())
    return nmae
