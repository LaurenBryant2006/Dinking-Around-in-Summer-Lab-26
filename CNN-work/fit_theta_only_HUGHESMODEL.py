# %%
import os
import time
import psutil
import numpy as np
import polars as pl
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import argparse
# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-tag", type=str, required=True)
parser.add_argument("-output_model_path", type=str, required=True)
args = parser.parse_args()
tag = args.tag
output_model_path = args.output_model_path
print(f"tag for this run: {tag}")
# --- CONFIGURATION ---
PATHS = {
    "train": '/fs/ess/PAS2159/HughesLab2/ANITA_DATA/parquet_for_fitting/processed_training_data.parquet',
    "test": '/fs/ess/PAS2159/HughesLab2/ANITA_DATA/parquet_for_fitting/processed_testing_data.parquet'
}

# Metadata features toggle
# META_FEATURES = [
#     'snr_calc_vpol_scaled', 
#     'snr_calc_hpol_scaled', 
#     'snr_dig_v_scaled', 
#     'snr_dig_h_scaled'
# ]
# Metadata features toggle
META_FEATURES = [
    'snr_dig_v_scaled', 
    'snr_dig_h_scaled',
    'snr_calc_hpol_over_vpol', 
    'trigger_ratio_pol'
]

TIME_BINS = 100
PI_180 = np.pi / 180
THETA_MIN = 5.664146440187189
THETA_MAX = 14.999993638611265
THETA_RANGE = THETA_MAX - THETA_MIN


# --- UTILITY FUNCTIONS ---

def printmem(text=''):
    """Reports current RSS memory usage."""
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    prefix = f"{text}; " if text else ""
    print(f"{prefix}Memory usage: {mem_mb:.2f} MB")

def reconstruct_images(df):
    """Reshapes flattened list column back to (N, 48, 100, 2) array."""
    # np.stack is the fastest way to turn a Polars series of lists into a 2D array
    flattened = np.stack(df["processed_waveforms"].to_numpy())
    return flattened.reshape(-1, 48, TIME_BINS, 2)

def finalize_arrays(df, meta_cols, scaling_params=None):
    """Extracts features/targets and applies min-max scaling to theta."""
    x_snr = df.select(meta_cols).to_numpy()
    y = df.select(['sin_phi', 'cos_phi', 'mctheta']).to_numpy()
    
    # Calculate or apply elevation scaling constants
    if scaling_params is None:
        params = {
            'theta_min': float(y[:, 2].min()), 
            'theta_max': float(y[:, 2].max())
        }
    else:
        params = scaling_params

    # Standard Min-Max scaling for elevation (index 2)
    y[:, 2] = (y[:, 2] - params['theta_min']) / (params['theta_max'] - params['theta_min'])
    
    return x_snr, y, params

# --- MAIN EXECUTION ---

printmem("Initial state")
t_start = time.time()

print("\n1. Loading pre-processed Parquet files...")
df_train = pl.read_parquet(PATHS["train"])
df_test = pl.read_parquet(PATHS["test"])

print("2. Reconstructing 4D CNN image arrays...")
x_train_img = reconstruct_images(df_train)
x_test_img = reconstruct_images(df_test)

print(f"3. Finalizing arrays using features: {META_FEATURES}")
x_train_snr, y_train, train_params = finalize_arrays(df_train, META_FEATURES)
x_test_snr, y_test, _ = finalize_arrays(df_test, META_FEATURES, scaling_params=train_params)
print(f"params {train_params}")

# Final stats
total_time = time.time() - t_start
print(f"\n>>> Setup Complete in {total_time:.2f}s")
print(f">>> Training X_img shape: {x_train_img.shape}")
print(f">>> Training Targets shape: {y_train.shape}")
printmem("Final state")


# Assuming your weight column in the parquet is named 'weight'
weights_train = df_train.select('weight').to_numpy().flatten()
weights_test = df_test.select('weight').to_numpy().flatten()

# print max/min of weights_train and weights_test
print(f"Max weight train: {weights_train.max():.2f}, Min weight train: {weights_train.min():.2e}")
print(f"Max weight test: {weights_test.max():.2f}, Min weight test: {weights_test.min():.2e}")
# Normalization: This ensures the global learning rate remains stable 
# while maintaining the relative importance of each event.

print('WEIGHTS ARE NOT NORMALIZED')
# weights_train = weights_train / np.mean(weights_train)
# weights_test = weights_test / np.mean(weights_test)
print(f"Weights normalized. Max: {weights_train.max():.2f}, Min: {weights_train.min():.2e}")
print(f"Max weight train: {weights_train.max():.2f}, Min weight train: {weights_train.min():.2e}")
print(f"Max weight test: {weights_test.max():.2f}, Min weight test: {weights_test.min():.2e}")
import matplotlib.pyplot as plt
import numpy as np

def plot_weight_diagnostics(df_train, df_test):
    # Extract weights
    w_train = df_train.select('weight').to_numpy().flatten()
    w_test = df_test.select('weight').to_numpy().flatten()

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Linear Scale Histogram
    ax[0].hist(w_train, bins=100, alpha=0.7, label=f'Train (N={len(w_train)})', color='steelblue', density=True)
    ax[0].hist(w_test, bins=100, alpha=0.5, label=f'Test (N={len(w_test)})', color='orange', density=True)
    ax[0].set_title("Weight Distribution (Linear Scale)")
    ax[0].set_xlabel("Weight Value")
    ax[0].set_ylabel("Density")
    ax[0].legend()
    ax[0].grid(axis='y', alpha=0.3)

    # 2. Log-Log Scale Histogram
    # We create log-spaced bins to properly see the 10^-6 to 1.0 range
    log_bins = np.logspace(np.log10(max(w_train.min(), 1e-7)), np.log10(w_train.max()), 100)
    
    ax[1].hist(w_train, bins=log_bins, alpha=0.7, label='Train', color='steelblue', density=True)
    ax[1].hist(w_test, bins=log_bins, alpha=0.5, label='Test', color='orange', density=True)
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_title("Weight Distribution (Log-Log Scale)")
    ax[1].set_xlabel("Weight Value (Log10)")
    ax[1].set_ylabel("Density (Log10)")
    ax[1].legend()
    ax[1].grid(which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"figs/weight_distribution_{tag}.png")

    if False:
        plt.show()

    # Calculate Effective Sample Size (ESS)
    # ESS = (sum w)^2 / sum (w^2)
    ess_train = (np.sum(w_train)**2) / np.sum(w_train**2)
    print(f"--- Weight Statistics ---")
    print(f"Mean weight: {np.mean(w_train):.2e}")
    print(f"Median weight: {np.median(w_train):.2e}")
    print(f"Max/Min ratio: {w_train.max() / w_train.min():.2e}")
    print(f"Effective Sample Size (ESS): {ess_train:.1f} (out of {len(w_train)} samples)")
    print(f"Efficiency: {100 * ess_train / len(w_train):.2f}%")

# Call the function
plot_weight_diagnostics(df_train, df_test)


import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# --- 1. CUSTOM LAYER & METRICS ---

@tf.keras.utils.register_keras_serializable(package="Custom")
class CircularPadding(layers.Layer):
    """Slices the last and first antennas to create a wrap-around effect."""
    def __init__(self, axis=1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        # inputs shape: (Batch, Antennas, Time, Pol)
        last_row = inputs[:, -1:, :, :]
        first_row = inputs[:, :1, :, :]
        return tf.concat([last_row, inputs, first_row], axis=self.axis)

    def compute_output_shape(self, input_shape):
        new_shape = list(input_shape)
        if new_shape[1] is not None:
            new_shape[1] += 2
        return tuple(new_shape)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config

def theta_metric_split(y_true, y_pred):
    # theta_range is used to de-scale the 0-1 target back to degrees for the metric
    theta_range = 14.999993638611265 - 5.664146440187189
    return tf.math.abs(y_true - y_pred) * theta_range

# --- 2. MODEL CONSTRUCTORS ---

def build_theta_simple(time_bins=100, activation='relu'):
    """Standard CNN architecture for Elevation (Theta)."""
    img_in = layers.Input(shape=(48, time_bins, 2), name="img_input")
    snr_in = layers.Input(shape=(4,), name="snr_input")
    
    # Spatial Processing
    x = CircularPadding(axis=1)(img_in)
    x = layers.Conv2D(32, (3, 3), activation=activation)(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation=activation)(x)
    x = layers.Flatten()(x)
    
    # Metadata Processing
    y = layers.Dense(16, activation=activation)(snr_in)
    
    # Fusion
    merged = layers.Concatenate()([x, y])
    z = layers.Dense(64, activation=activation)(merged)
    z = layers.Dense(32, activation=activation)(z)
    
    out = layers.Dense(1, activation='linear', name="theta_out")(z)
    
    model = models.Model(inputs=[img_in, snr_in], outputs=out, name="Theta_Simple")
    model.compile(optimizer='adam', loss='mse', metrics=[theta_metric_split])
    return model

def build_theta_sharpener_v2(time_bins=100, activation='relu'):
    """Deeper architecture using TimeDistributed Conv1D for pulse detection."""
    img_in = layers.Input(shape=(48, time_bins, 2), name="img_input")
    snr_in = layers.Input(shape=(4,), name="snr_input")

    # Waveform Sharpening (Filters time-bins per antenna)
    x = layers.TimeDistributed(layers.Conv1D(16, kernel_size=7, padding='same', activation=activation))(img_in)
    x = layers.Reshape((48, time_bins, 16))(x)
    
    # Spatial Integration
    x = CircularPadding(axis=1)(x)
    x = layers.Conv2D(32, (3, 3), activation=activation)(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation=activation)(x)
    x = layers.Flatten()(x)
    
    # Metadata Processing
    y = layers.Dense(16, activation=activation)(snr_in)
    
    # Fusion
    merged = layers.Concatenate()([x, y])
    z = layers.Dense(64, activation=activation)(merged)
    z = layers.Dense(32, activation=activation)(z)
    
    out = layers.Dense(1, activation='linear', name="theta_out")(z)
    
    model = models.Model(inputs=[img_in, snr_in], outputs=out, name="Theta_Sharpener")
    model.compile(optimizer='adam', loss='mse', metrics=[theta_metric_split])
    return model

def build_theta_sharpener_v2a(time_bins=100, activation='relu'):
    img_in = layers.Input(shape=(48, time_bins, 2), name="img_input")
    meta_in = layers.Input(shape=(4,), name="snr_input")

    # 1. Multi-Scale Waveform Sharpening (Temporal)
    # Keeping it simple: Small (3) and Medium (7)
    h1 = layers.TimeDistributed(layers.Conv1D(8, 3, padding='same', activation=activation))(img_in)
    h2 = layers.TimeDistributed(layers.Conv1D(8, 7, padding='same', activation=activation))(img_in)
    x = layers.Concatenate()([h1, h2]) 
    x = layers.BatchNormalization()(x) # Balance the two temporal scales
    x = layers.Reshape((48, time_bins, 16))(x)
    
    # 2. Spatial Integration
    # Padding twice to handle larger kernels (like 5x5)
    x = CircularPadding(axis=1)(x)
    x = CircularPadding(axis=1)(x)
    
    # Parallel Spatial Heads
    s1 = layers.Conv2D(16, (3, 3), padding='valid', activation=activation)(x)
    s2 = layers.Conv2D(16, (5, 5), padding='valid', activation=activation)(x)
    # Use 'valid' because CircularPadding handled the edges manually
    
    # We need to make sure shapes match to concatenate if they aren't 'same'
    # Or just use 'same' for simplicity:
    s1 = layers.Conv2D(16, (3, 3), padding='same', activation=activation)(x)
    s2 = layers.Conv2D(16, (5, 5), padding='same', activation=activation)(x)
    
    x = layers.Concatenate()([s1, s2])
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # --- CRITICAL STEP: The Bottleneck ---
    # Reduce the 32 filters down to 8 before flattening to save parameters
    x = layers.Conv2D(8, (1, 1), activation=activation)(x) 
    
    x = layers.Flatten()(x)
    
    # 3. Metadata Fusion
    y = layers.Dense(16, activation=activation)(meta_in)
    
    merged = layers.Concatenate()([x, y])
    z = layers.Dense(64, activation=activation)(merged)
    z = layers.Dense(32, activation=activation)(z)
    
    out = layers.Dense(1, activation='linear', name="theta_out")(z)

    model = models.Model(inputs=[img_in, meta_in], outputs=out)
    model.compile(optimizer='adam', loss='mse', metrics=[theta_metric_split])
    return model

def plot_training_history(history):
    # Extract data from history object
    acc = history.history['theta_metric_split']
    val_acc = history.history['val_theta_metric_split']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(1, len(loss) + 1)
    # Get the minimum value of the validation loss for train and test
    min_val_loss = min(val_loss)
    min_val_loss_index = val_loss.index(min_val_loss)
    min_val_loss_epoch = epochs_range[min_val_loss_index]
    print(f"Minimum validation loss: {min_val_loss} at epoch {min_val_loss_epoch}")

    plt.figure(figsize=(15, 5))

    # Plot 1: Loss (MSE)
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, loss, label='Training Loss (Weighted)')
    plt.plot(epochs_range, val_loss, label='Validation Loss (Weighted)')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    # Plot 2: Metric (Degrees)
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, acc, label='Training Angular Error')
    plt.plot(epochs_range, val_acc, label='Validation Angular Error')
    plt.title('Training and Validation Theta Metric')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error (Degrees)')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    # add a vertical line at the minimum validation loss epoch
    plt.axvline(x=min_val_loss_epoch, color='red', linestyle='--', label=f'Minimum Validation Loss at Epoch {min_val_loss_epoch}')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    # add a horizontal line at the minimum validation loss
    plt.axhline(y=min_val_loss, color='red', linestyle='--', label=f'Minimum Validation Loss: {min_val_loss:.4f}')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"figs/theta_training_history_{tag}.png")
    if False:
        plt.show()

# 1. Move this to the top level (global scope) of your script
@tf.keras.utils.register_keras_serializable(package="Custom")
def apply_snr_attention(inputs):
    img, meta = inputs
    # v_weight = meta[:, 0], h_weight = meta[:, 1]
    v_w = tf.reshape(meta[:, 0], (-1, 1, 1, 1))
    h_w = tf.reshape(meta[:, 1], (-1, 1, 1, 1))
    h_chan = img[:, :, :, 0:1] * h_w
    v_chan = img[:, :, :, 1:2] * v_w
    return tf.concat([h_chan, v_chan], axis=-1)


@tf.keras.utils.register_keras_serializable(package="Custom")
class SectorPadding(layers.Layer):
    """Circular padding for the 16-sector azimuthal axis."""
    def __init__(self, **kwargs):
        super(SectorPadding, self).__init__(**kwargs)

    def call(self, inputs):
        # inputs: (Batch, Rings=3, Sectors=16, Time=100, Features=24)
        # Pad axis 2 (the 16 sectors)
        # Take the last sector and put it at the start, first sector at the end
        return tf.concat([inputs[:, :, -1:, :, :], inputs, inputs[:, :, :1, :, :]], axis=2)

    def compute_output_shape(self, input_shape):
        # input_shape: (None, 3, 16, 100, 24)
        # output_shape: (None, 3, 18, 100, 24)
        if input_shape[2] is None:
            return input_shape
        return (input_shape[0], input_shape[1], input_shape[2] + 2, input_shape[3], input_shape[4])

def build_theta_phased_array_v4(time_bins=100, activation='elu'):
    img_in = layers.Input(shape=(48, time_bins, 2), name="img_input")
    
    # --- CHANGE 1: Metadata Input shape is now 4 ---
    meta_in = layers.Input(shape=(4,), name="meta_input")
    
    # --- EARLY SNR WEIGHTING ---
    # def apply_snr_attention(inputs):
    #     img, meta = inputs
    #     # Assuming meta[0] is V_SNR and meta[1] is H_SNR
    #     v_w = tf.reshape(meta[:, 0], (-1, 1, 1, 1))
    #     h_w = tf.reshape(meta[:, 1], (-1, 1, 1, 1))
    #     h_chan = img[:, :, :, 0:1] * h_w
    #     v_chan = img[:, :, :, 1:2] * v_w
    #     return tf.concat([h_chan, v_chan], axis=-1)

    # x = layers.Lambda(apply_snr_attention)([img_in, meta_in])
    x = layers.Lambda(apply_snr_attention, name="snr_weighting")([img_in, meta_in])
    # --- MULTI-SCALE TEMPORAL SHARPENING ---
    t1 = layers.TimeDistributed(layers.Conv1D(12, 3, padding='same', activation=activation))(x)
    t2 = layers.TimeDistributed(layers.Conv1D(12, 7, padding='same', dilation_rate=2, activation=activation))(x)
    x = layers.Concatenate()([t1, t2]) 
    x = layers.BatchNormalization()(x)
    
    # --- GEOMETRIC RESHAPE ---
    x = layers.Reshape((3, 16, time_bins, 24))(x)
    
    # Circular Padding on the 16-sector axis
#    x = layers.Lambda(lambda t: tf.concat([t[:, :, -1:, :, :], t, t[:, :, :1, :, :]], axis=2))(x)
# NEW CUSTOM LAYER (Use this):
    x = SectorPadding(name="sector_padding")(x)

    # --- CONV3D INTERFEROMETRY ---
    # We keep (3, 3, 5) because it captures the vertical rings and temporal slope.
    x = layers.Conv3D(32, (3, 3, 5), activation=activation, padding='valid')(x)
    
    # --- PARAMETER RECOVERY ---
    x = layers.Flatten()(x)
    
    # Metadata Fusion
    m = layers.Dense(16, activation=activation)(meta_in)
    merged = layers.Concatenate()([x, m])
    
    # Wide Dense layers to keep the parameter count high (~800k)
    z = layers.Dense(512, activation=activation)(merged) 
    z = layers.Dropout(0.2)(z)
    z = layers.Dense(128, activation=activation)(z)
    
    # --- CHANGE 2: Output is a single value (Theta) ---
    # We use a linear output because Theta is -11 to +11, not 0-360.
    theta_out = layers.Dense(1, activation='linear', name="theta_out")(z)

    model = models.Model(inputs=[img_in, meta_in], outputs=theta_out)
    
    # # Using Huber loss is often better for Theta to ignore noisy outliers
    # model.compile(
    #     optimizer=tf.keras.optimizers.Adam(learning_rate=0.0008), 
    #     loss='huber', 
    #     metrics=[theta_metric_split]
    # )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0008), 
                  loss='mse', metrics=[theta_metric_split])

    return model

# --- 3. TRAINING PIPELINE ---

# Configuration
EPOCHS = 100           # Increased from 5 for a real run
BATCH_SIZE = 32

# Prepare Targets (mctheta is index 2)
y_train_theta = y_train[:, 2:]
y_test_theta = y_test[:, 2:]

ACTIVATION = 'relu'
#ACTIVATION = 'elu'
# leaky relu
#ACTIVATION = 'leaky_relu'
print(f"Using activation function: {ACTIVATION}")
# Initialize chosen model

print("\n>>> Training Theta Specialist with Pulse Sharpener...build_theta_phased_array_v4")
theta_model = build_theta_phased_array_v4(time_bins=TIME_BINS, activation=ACTIVATION)
model_name = f"theta_specialist_sharpener_{ACTIVATION}"

theta_model.summary()

callbacks_list = [
    # Monitor the angular error instead of the MSE loss
    callbacks.EarlyStopping(
        monitor='val_theta_metric_split', 
        patience=15, 
        mode='min',            # We want to MINIMIZE the degrees of error
        restore_best_weights=True,
        verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_theta_metric_split', 
        factor=0.5, 
        patience=7, 
        mode='min', 
        verbose=1
    ),
    callbacks.ModelCheckpoint(
    filepath=output_model_path,
    monitor='val_theta_metric_split',          # You can also use 'val_phi_metric_split'
    save_best_only=True,         # Only save when the monitored value improves
    mode='min',                  # 'min' for loss/error, 'max' for accuracy
    verbose=1,                   # Prints a message when a new best is saved
    save_weights_only=False      # Saves the entire architecture + weights
)
]


# Training execution
t_train = time.time()
theta_history = theta_model.fit(
    x=[x_train_img, x_train_snr], 
    y=y_train_theta,
    sample_weight=weights_train,
    validation_data=([x_test_img, x_test_snr], y_test_theta, weights_test),
    epochs=EPOCHS, 
    batch_size=BATCH_SIZE,
    callbacks=callbacks_list
)

# --- 4. SAVING ---

# Save the model
# model_path = output_model_path
# theta_model.save(model_path)
# print(f"Model saved to {model_path}")

# Save the scaling parameters
params_path = f"/fs/ess/PAS3311/anita/models/train_params_{model_name}.json"
with open(params_path, 'w') as f:
    json.dump(train_params, f)

print(f"Training parameters saved to {params_path}")
print(f"Training Time: {time.time()-t_train:.2f} seconds")

#print(f"Training Time: {time.time()-t_train:.2f} seconds")
# Call the function after phi_model.fit()
plot_training_history(theta_history)


# 1. Generate predictions from BOTH models
# Each model takes the same inputs but produces a specific output
#
# Add to the train dataframe
theta_preds = theta_model.predict([x_train_img, x_train_snr]) # Shape: (N, 1)
theta_range = train_params['theta_max'] - train_params['theta_min']
THETA_MIN, THETA_MAX = train_params['theta_min'], train_params['theta_max']
theta_pred_deg = (theta_preds.flatten() * theta_range) + train_params['theta_min']
theta_true_deg = (y_train_theta.flatten() * theta_range) + train_params['theta_min']

df_train = df_train.with_columns(pl.Series(theta_pred_deg).alias("theta_pred_deg"))
df_train = df_train.with_columns(pl.Series(theta_true_deg).alias("theta_true_deg"))
#
# Only save with columns run, event, theta_pred_deg, theta_true_deg
df_train = df_train.select(['run', 'event', 'theta_pred_deg', 'theta_true_deg',
'theta_estimated','maxSNRAtTriggerH','maxSNRAtTriggerV','weight'])
df_train.write_csv(f"/fs/ess/PAS3311/anita/data/train_theta_only_predictions_model_{tag}.csv")

#
# Now do test
theta_preds = theta_model.predict([x_test_img, x_test_snr]) # Shape: (N, 1)
theta_range = train_params['theta_max'] - train_params['theta_min']

theta_pred_deg = (theta_preds.flatten() * theta_range) + train_params['theta_min']
theta_true_deg = (y_test_theta.flatten() * theta_range) + train_params['theta_min']

# Add to the test dataframe
df_test = df_test.with_columns(pl.Series(theta_pred_deg).alias("theta_pred_deg"))
df_test = df_test.with_columns(pl.Series(theta_true_deg).alias("theta_true_deg"))
#
# Only save with columns run, event, theta_pred_deg, theta_true_deg
df_test = df_test.select(['run', 'event', 'theta_pred_deg', 'theta_true_deg',
'theta_estimated','maxSNRAtTriggerH','maxSNRAtTriggerV','weight'])
df_test.write_csv(f"/fs/ess/PAS3311/anita/data/test_theta_only_predictions_model_{tag}.csv")

# Calculate absolute error in degrees
theta_error = np.abs(theta_true_deg - theta_pred_deg)
print(f"Mean Absolute Error MLAND Delta Theta: {np.mean(theta_error):.2f}°")
print(f"Standard Deviation of MLAND Delta Theta: {np.std(theta_error):.2f}°")

# Ensure weights_test is a 1D array to match the error arrays
# If weights_test was a DataFrame column or a (N, 1) array, flatten it.
weights = np.asarray(weights_test).flatten()

# --- Weighted Stats for Theta ---
w_mae_theta = np.average(theta_error, weights=weights)
w_var_theta = np.average((theta_error - w_mae_theta)**2, weights=weights)
w_std_theta = np.sqrt(w_var_theta)

print("-" * 30)
print(f"WEIGHTED STATS (using test weights):")
print(f"Weighted MAE Theta: {w_mae_theta:.2f}°")
print(f"Weighted Std Theta: {w_std_theta:.2f}°")
print("-" * 30)

from scipy.optimize import curve_fit
import numpy as np
# 1. Define the Gaussian function
def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean)**2) / (2 * stddev**2))

from scipy.optimize import curve_fit

from scipy.optimize import curve_fit
def plot_gaussian_fit(data_series, ax, weights=None, bins=100, fit_range=(-10, 10), title="Histogram Fit", xlabel="Degrees"):
    """
    Fits a Gaussian to data (weighted or unweighted) and plots it.
    """
    data = np.asarray(data_series)
    mask = (data >= fit_range[0]) & (data <= fit_range[1])
    
    fit_data = data[mask]
    fit_weights = weights[mask] if weights is not None else None

    if len(fit_data) < 10:
        ax.set_title(f"{title} (Insufficient Data)")
        return None

    # 1. Prepare histogram for fitting
    # If weights is provided, counts becomes the sum of weights in each bin
    counts, bin_edges = np.histogram(fit_data, bins=bins, range=fit_range, weights=fit_weights)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 2. Initial guess [Amplitude, Mean, StdDev]
    if fit_weights is not None:
        weighted_mean = np.average(fit_data, weights=fit_weights)
        weighted_std = np.sqrt(np.average((fit_data - weighted_mean)**2, weights=fit_weights))
        p0 = [counts.max(), weighted_mean, weighted_std]
    else:
        p0 = [counts.max(), np.mean(fit_data), np.std(fit_data)]

    # 3. Gaussian Fit
    def gaussian(x, amp, mean, sigma):
        return amp * np.exp(-((x - mean)**2) / (2 * sigma**2))

    try:
        popt, _ = curve_fit(gaussian, bin_centers, counts, p0=p0)
        amp, mean, sigma = popt
    except Exception as e:
        ax.set_title(f"{title} (Fit Failed)")
        return None

    # 4. Plotting
    color = 'salmon' if weights is not None else 'skyblue'
    label_suffix = " (Weighted)" if weights is not None else ""
    
    ax.hist(data, bins=bins, range=fit_range, weights=weights, 
            color=color, edgecolor='black', alpha=0.7, label=f'Data{label_suffix}')
    
    x_plot = np.linspace(fit_range[0], fit_range[1], 1000)
    ax.plot(x_plot, gaussian(x_plot, *popt), color='red', lw=2.5, label='Gaussian Fit')

    # 5. Formatting & Stats
    stats_text = (f"$\mu$: {mean:.3f}\n"
                  f"$\sigma$: {abs(sigma):.3f}\n"
                  f"Amp: {amp:.1f}")
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=12, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Sum of Weights" if weights is not None else "Counts")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    return popt

# Create a 2x3 grid for Theta: Top = Standard, Bottom = Weighted
fig, ax = plt.subplots(2, 3, figsize=(18, 12))
weights_array = np.asarray(weights_test).flatten()

# --- 1. Calculate Residuals ---
# MLAND Residuals
delta_theta_mland = theta_pred_deg - theta_true_deg

# ANITA Residuals (matching your logic from before)
# Note: Ensure the signs/math match your specific dataset convention here
delta_theta_anita = df_test['theta_estimated'].to_numpy() + theta_true_deg.flatten()

# --- ROW 0: STANDARD (Unweighted) ---
# Hexbin Comparison
hb0 = ax[0,0].hexbin(theta_true_deg, theta_pred_deg, gridsize=50, cmap='inferno', mincnt=1)
ax[0,0].plot([THETA_MIN, THETA_MAX], [THETA_MIN, THETA_MAX], 'w--', alpha=0.5)
ax[0,0].set_title("Theta Performance (Unweighted)")
ax[0,0].set_xlabel("True Theta (deg)")
ax[0,0].set_ylabel("Predicted Theta (deg)")
fig.colorbar(hb0, ax=ax[0,0], label='Counts')

# MLAND Gaussian Fit
plot_gaussian_fit(delta_theta_mland, ax[0,1], 
                  fit_range=(-2, 2), 
                  title="MLAND Residual (Unweighted)",
                  xlabel="$\Delta\\theta$ (deg)")

# ANITA Gaussian Fit
plot_gaussian_fit(delta_theta_anita, ax[0,2], 
                  fit_range=(-2, 2), 
                  title="ANITA Residual (Unweighted)",
                  xlabel="$\Delta\\theta$ (deg)")

# --- ROW 1: WEIGHTED ---
# Weighted Hexbin Comparison
# C=weights_array with np.sum means color = total weight in that bin
hb1 = ax[1,0].hexbin(theta_true_deg, theta_pred_deg, C=weights_array, reduce_C_function=np.sum, 
                     gridsize=50, cmap='viridis', mincnt=1)
ax[1,0].plot([THETA_MIN, THETA_MAX], [THETA_MIN, THETA_MAX], 'w--', alpha=0.5)
ax[1,0].set_title("Theta Performance (Weighted)")
ax[1,0].set_xlabel("True Theta (deg)")
ax[1,0].set_ylabel("Predicted Theta (deg)")
fig.colorbar(hb1, ax=ax[1,0], label='Sum of Weights')

# MLAND Weighted Gaussian Fit
plot_gaussian_fit(delta_theta_mland, ax[1,1], weights=weights_array, 
                  fit_range=(-2, 2), 
                  title="MLAND Residual (Weighted)",
                  xlabel="$\Delta\\theta$ (deg)")

# ANITA Weighted Gaussian Fit
plot_gaussian_fit(delta_theta_anita, ax[1,2], weights=weights_array, 
                  fit_range=(-2, 2), 
                  title="ANITA Residual (Weighted)",
                  xlabel="$\Delta\\theta$ (deg)")

#plt.tight_layout()
#plt.savefig(f"figs/int_theta_only_weighted_comparison_{tag}.png")

if False:
    plt.show()

# Final clean up
plt.tight_layout()
plt.savefig(f"figs/theta_performance_side_by_side_model_{tag}.png")
if False:
    plt.show()  

#
print("done and successfully saved the theta model and the predictions!")