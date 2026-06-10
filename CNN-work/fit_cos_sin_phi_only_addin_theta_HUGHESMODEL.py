##Notes for myself regarding Dr.Hughes CNN Model
## The theta model in the other file is trained first then the phi model re-uses the trained theta model's predictions as an extra input feature
## This is far far more complicated than anything that I have done in my first CNN model
## he uses rings x sectors, uses SNR metadata to weight the polarization channels before convolution
## [Train theta model]  →  [Use its predictions to help train phi model]

# %%
import os
import time
import psutil
import json
import numpy as np
import polars as pl
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
from scipy.stats import norm
import argparse
# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-tag", type=str, required=True)
parser.add_argument("-input_theta_model_path", type=str, required=True)
parser.add_argument("-output_model_path", type=str, required=True)
args = parser.parse_args()
tag = args.tag
output_model_path = args.output_model_path
input_theta_model_path = args.input_theta_model_path
print(f"tag for this run: {tag}")


# --- CONFIGURATION ---
PATHS = {
    "train": '/fs/ess/PAS2159/HughesLab2/ANITA_DATA/parquet_for_fitting/processed_training_data.parquet',
    "test": '/fs/ess/PAS2159/HughesLab2/ANITA_DATA/parquet_for_fitting/processed_testing_data.parquet',
}
THETA_MIN = 5.664146440187189
THETA_MAX = 14.999993638611265
THETA_RANGE = THETA_MAX - THETA_MIN

META_FEATURES = [
    'snr_dig_v_scaled', 
    'snr_dig_h_scaled',
    'snr_calc_hpol_over_vpol', 
    'trigger_ratio_pol'
]

# META_FEATURES = ['snr_calc_vpol_scaled', 'snr_calc_hpol_scaled', 'snr_dig_v_scaled', 'snr_dig_h_scaled']
TIME_BINS = 100
PI_180 = np.pi / 180
ACTIVATION = 'relu'
#ACTIVATION = 'elu'
# leaky relu
#ACTIVATION = 'leaky_relu'
print(f"Using activation function: {ACTIVATION}")
# --- 1. CUSTOM LAYER & METRIC ---
# 1. Move this to the top level (global scope) of your script

# Named function for normalization to avoid Lambda errors
@tf.keras.utils.register_keras_serializable(package="Custom")
def l2_norm(t):
    return tf.math.l2_normalize(t, axis=1)

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
class CircularPadding(layers.Layer):
    """Slices the last and first antennas to create a wrap-around effect."""
    def __init__(self, axis=1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        last_row = inputs[:, -1:, :, :]
        first_row = inputs[:, :1, :, :]
        return tf.concat([last_row, inputs, first_row], axis=self.axis)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config

@tf.keras.utils.register_keras_serializable(package="Custom")
def phi_metric_split(y_true, y_pred):
    """Angular distance metric for [sin, cos] targets."""
    true_rad = tf.math.atan2(y_true[:, 0], y_true[:, 1])
    pred_rad = tf.math.atan2(y_pred[:, 0], y_pred[:, 1])
    # Shortest distance on a circle
    delta_rad = tf.math.atan2(tf.math.sin(true_rad - pred_rad), tf.math.cos(true_rad - pred_rad))
    return tf.math.abs(delta_rad) * (180.0 / np.pi)

# Map the names Keras saved to the actual functions/classes in this script
custom_dict = {
    "CircularPadding": CircularPadding,
    "apply_snr_attention": apply_snr_attention,
    "phi_metric_split": phi_metric_split,
    "function": apply_snr_attention # Sometimes Keras registers it under 'function'
}

# --- 2. UTILITY FUNCTIONS ---

def printmem(text=''):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    print(f"{text + '; ' if text else ''}Memory usage: {mem_mb:.2f} MB")

def reconstruct_images(df):
    flattened = np.stack(df["processed_waveforms"].to_numpy())
    return flattened.reshape(-1, 48, TIME_BINS, 2)

# --- 3. DATA LOADING ---

printmem("Initial state")
print("1. Loading pre-processed Parquet files...")
df_train = pl.read_parquet(PATHS["train"])
df_test = pl.read_parquet(PATHS["test"])

print("Columns in df_train: ", df_train.columns)
print("Columns in df_test: ", df_test.columns)

print("2. Reconstructing CNN image arrays...")
x_train_img = reconstruct_images(df_train)
x_test_img = reconstruct_images(df_test)

# Extract basic SNR and [sin, cos] targets
# No scaling needed here as sin/cos are already -1 to 1
x_train_snr = df_train.select(META_FEATURES).to_numpy()
x_test_snr = df_test.select(META_FEATURES).to_numpy()
y_train_phi = df_train.select(['sin_phi', 'cos_phi']).to_numpy()
y_test_phi = df_test.select(['sin_phi', 'cos_phi']).to_numpy()

# --- 4. THETA HINT GENERATION ---

print(f"3. Loading Theta model to generate hints from: {input_theta_model_path}")
# Load theta model with custom objects

def theta_metric_split(y_true, y_pred):
    # theta_range is used to de-scale the 0-1 target back to degrees for the metric
    return tf.math.abs(y_true - y_pred) * THETA_RANGE

# theta_model = tf.keras.models.load_model(
#     input_theta_model_path, 
#     custom_objects=custom_dict
# #    custom_objects={"CircularPadding": CircularPadding, "theta_metric_split": theta_metric_split} 
# )
# Add this line to bypass the security check
tf.keras.config.enable_unsafe_deserialization()

# Now your existing load call will work
theta_model = tf.keras.models.load_model(
    input_theta_model_path,
    custom_objects={
        "SectorPadding": SectorPadding,
        "apply_snr_attention": apply_snr_attention,
        "l2_norm": l2_norm,
        "theta_metric_split": theta_metric_split
    }
)

print("4. Generating Theta hints...")
# Theta model expects [images, snr_metadata]
theta_hints_train = theta_model.predict([x_train_img, x_train_snr], batch_size=256)
theta_hints_test = theta_model.predict([x_test_img, x_test_snr], batch_size=256)


# Transform the hints to degrees - only for saving, since the model wants them scaled to 0-1
theta_hints_train_deg = (theta_hints_train.flatten() * THETA_RANGE) + THETA_MIN
theta_hints_test_deg = (theta_hints_test.flatten() * THETA_RANGE) + THETA_MIN


# Append hint to metadata: (N, 4) -> (N, 5)
# Reshape the hints to be (N, 1) instead of (N,)
x_train_phi_meta = np.hstack([x_train_snr, theta_hints_train.reshape(-1, 1)])
x_test_phi_meta = np.hstack([x_test_snr, theta_hints_test.reshape(-1, 1)])



# Assuming your weight column in the parquet is named 'weight'
weights_train = df_train.select('weight').to_numpy().flatten()
weights_test = df_test.select('weight').to_numpy().flatten()


print(f"New Metadata Shape: {x_train_phi_meta.shape}")

# --- 5. MODEL ARCHITECTURE ---

def build_phi_pulse_sharpener(time_bins=100, activation='relu'):
    img_in = layers.Input(shape=(48, time_bins, 2), name="img_input")
    meta_in = layers.Input(shape=(5,), name="meta_input") # 4 SNRs + 1 Theta Hint
    
    # Waveform Sharpening
    x = layers.TimeDistributed(layers.Conv1D(16, kernel_size=7, padding='same', activation=activation))(img_in)
    x = layers.Reshape((48, time_bins, 16))(x)
    
    # Spatial Integration
    x = CircularPadding(axis=1)(x)
    x = layers.Conv2D(32, (3, 3), activation=activation)(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation=activation)(x)
    x = layers.Flatten()(x)
    
    # Metadata Fusion
    y = layers.Dense(16, activation=activation)(meta_in)
    
    merged = layers.Concatenate()([x, y])
    z = layers.Dense(64, activation=activation)(merged)
    z = layers.Dense(32, activation=activation)(z)
    
    # Output is sin and cos of Phi
    out = layers.Dense(2, activation='linear', name="phi_out")(z)
    
    model = models.Model(inputs=[img_in, meta_in], outputs=out, name="Phi_Pulse_Sharpener")
    model.compile(optimizer='adam', loss='mse', metrics=[phi_metric_split])
    return model

def build_phi_pulse_sharpener_v2(time_bins=100, activation='relu'):
    img_in = layers.Input(shape=(48, time_bins, 2), name="img_input")
    meta_in = layers.Input(shape=(5,), name="meta_input")
    
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
    
    # Output with Unit Normalization
    z = layers.Dense(2, activation='linear')(z)
    out = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1), name="phi_out")(z)

    model = models.Model(inputs=[img_in, meta_in], outputs=out)
    model.compile(optimizer='adam', loss='mse', metrics=[phi_metric_split])
    return model
import tensorflow as tf
from tensorflow.keras import layers, models

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


def build_phi_phased_array_v4(time_bins=100, activation='elu'):
    img_in = layers.Input(shape=(48, time_bins, 2), name="img_input")
    meta_in = layers.Input(shape=(5,), name="meta_input")

    #x = layers.Lambda(apply_snr_attention)([img_in, meta_in])
    x = layers.Lambda(apply_snr_attention, name="snr_weighting")([img_in, meta_in])

    # --- 2. MULTI-SCALE TEMPORAL SHARPENING ---
    # Kernel 3 and 7 (Dilation 2) to capture sub-bin phase
    t1 = layers.TimeDistributed(layers.Conv1D(12, 3, padding='same', activation=activation))(x)
    t2 = layers.TimeDistributed(layers.Conv1D(12, 7, padding='same', dilation_rate=2, activation=activation))(x)
    x = layers.Concatenate()([t1, t2]) # 24 filters
    x = layers.BatchNormalization()(x)
    
    # --- 3. GEOMETRIC INTERFEROMETRY (Conv2D) ---
    # Reshape to (Batch, Rings=3, Sectors=16, Time=100, Features=24)
    # We treat Rings and Sectors as a 2D image for high-speed Conv2D
    x = layers.Reshape((3, 16, time_bins, 24))(x)
    
    # Circular Padding on the 16-sector axis (axis 2)
    #x = layers.Lambda(lambda t: tf.concat([t[:, :, -1:, :, :], t, t[:, :, :1, :, :]], axis=2))(x)
    x = SectorPadding(name="sector_padding")(x)
    # Process spatial features. Kernel (3, 3) sees all 3 rings and 3 sectors at once.
    x = layers.Conv3D(32, (3, 3, 5), activation=activation, padding='valid')(x)
    # Resulting shape is roughly (1, 14, 96, 32)
    
    # --- 4. THE PARAMETER RECOVERY (Flatten vs Pooling) ---
    # We DO NOT use GlobalPooling here. We Flatten to keep the parameters high.
    x = layers.Flatten()(x) # This will produce ~10k-15k neurons
    
    # Metadata Fusion
    m = layers.Dense(16, activation=activation)(meta_in)
    merged = layers.Concatenate()([x, m])
    
    # Wide Dense layers to reach ~800k - 1M parameters
    # This is where the model "stores" the 0.3-degree precision logic
    z = layers.Dense(512, activation=activation)(merged) 
    z = layers.Dropout(0.2)(z)
    z = layers.Dense(128, activation=activation)(z)
    
    # Unit Circle Output
    # out_raw = layers.Dense(2, activation='linear')(z)
    # out = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1), name="phi_out")(out_raw)
# 5. Output - USE THE NAMED l2_norm FUNCTION, NOT A LAMBDA
    out_raw = layers.Dense(2, activation='linear')(z)
    out = layers.Lambda(l2_norm, name="phi_out")(out_raw)

    model = models.Model(inputs=[img_in, meta_in], outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0008), 
                  loss='mse', metrics=[phi_metric_split])
    return model

# --- 6. TRAINING ---
EPOCHS = 100           # Increased from 5 for a real run
BATCH_SIZE = 32

print("\n>>> Training Phi Specialist with Theta Hints... using build_phi_phased_array_v4")
phi_model = build_phi_phased_array_v4(time_bins=TIME_BINS, activation=ACTIVATION)
phi_model.summary()

# Define the file path (use .keras extension for modern Keras)
# checkpoint_path = 'models/best_phased_array_v4.keras'

#monitor_metric = 'val_phi_metric_split'
monitor_metric = 'val_loss'

checkpoint_callback = callbacks.ModelCheckpoint(
    filepath=output_model_path,
    monitor=monitor_metric,          # You can also use 'val_phi_metric_split'
    save_best_only=True,         # Only save when the monitored value improves
    mode='min',                  # 'min' for loss/error, 'max' for accuracy
    verbose=1,                   # Prints a message when a new best is saved
    save_weights_only=False      # Saves the entire architecture + weights
)

callbacks_list = [
    callbacks.EarlyStopping(monitor=monitor_metric, patience=15, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor=monitor_metric, factor=0.5, patience=7, verbose=1),
    checkpoint_callback
]

t_train = time.time()
phi_history = phi_model.fit(
    x=[x_train_img, x_train_phi_meta], 
    y=y_train_phi,
    sample_weight=weights_train, 
    validation_data=([x_test_img, x_test_phi_meta], y_test_phi, weights_test),
    epochs=EPOCHS, 
    batch_size=BATCH_SIZE,
    callbacks=callbacks_list
)

print(f"Training Time: {time.time()-t_train:.2f} seconds")

# --- 7. SAVING ---
# phi_model.save(output_model_path)
# print(f"Phi Model saved successfully to {output_model_path}")



# 1. Generate predictions from BOTH models
# Each model takes the same inputs but produces a specific output
#
# Add to the train dataframe

# 1. Generate predictions from BOTH models
# Each model takes the same inputs but produces a specific output
phi_preds = phi_model.predict([x_train_img, x_train_phi_meta])     # Shape: (N, 2)

# 2. Extract and Transform Phi
# phi_preds[:, 0] is sin, phi_preds[:, 1] is cos
phi_pred_rad = np.arctan2(phi_preds[:, 0], phi_preds[:, 1])

# Using the sliced true values we created earlier (y_test_phi)
phi_true_rad = np.arctan2(y_train_phi[:, 0], y_train_phi[:, 1])

# Convert to degrees
phi_pred_deg = np.degrees(phi_pred_rad)
phi_true_deg = np.degrees(phi_true_rad)

df_train = df_train.with_columns(pl.Series(phi_pred_deg).alias("phi_pred_deg"))
df_train = df_train.with_columns(pl.Series(phi_true_deg).alias("phi_true_deg"))
df_train = df_train.with_columns(pl.Series(theta_hints_train_deg).alias("theta_hints_pred"))
#
# Only save with columns 
df_train = df_train.select(['run', 'event', 'phi_pred_deg', 'mcphi',
                            'theta_hints_pred','phi_estimated',
                            'maxSNRAtTriggerH','maxSNRAtTriggerV'
                            ])
df_train.write_csv(f"/fs/ess/PAS3311/anita/data/train_phi_only_predictions_model_{tag}.csv")

# Now test data

# 1. Generate predictions from BOTH models
# Each model takes the same inputs but produces a specific output
phi_preds = phi_model.predict([x_test_img, x_test_phi_meta])     # Shape: (N, 2)

# 2. Extract and Transform Phi
# phi_preds[:, 0] is sin, phi_preds[:, 1] is cos
phi_pred_rad = np.arctan2(phi_preds[:, 0], phi_preds[:, 1])

# Using the sliced true values we created earlier (y_test_phi)
phi_true_rad = np.arctan2(y_test_phi[:, 0], y_test_phi[:, 1])

# Convert to degrees
phi_pred_deg = np.degrees(phi_pred_rad)
phi_true_deg = np.degrees(phi_true_rad)

# Add to the test dataframe
df_test = df_test.with_columns(pl.Series(phi_pred_deg).alias("phi_pred_deg"))
df_test = df_test.with_columns(pl.Series(phi_true_deg).alias("phi_true_deg"))
df_test = df_test.with_columns(pl.Series(theta_hints_test_deg).alias("theta_hints_pred"))
#
# Only save with columns 
df_test = df_test.select(['run', 'event', 'phi_pred_deg', 'mcphi',
                            'theta_hints_pred','phi_estimated',
                            'maxSNRAtTriggerH','maxSNRAtTriggerV'
                            ])

df_test.write_csv(f"/fs/ess/PAS3311/anita/data/test_phi_only_predictions_model_{tag}.csv")


# Calculate absolute error in degrees
phi_error = np.abs(phi_true_deg - phi_pred_deg)
# Handle the wrap-around (e.g., error between 179 and -179 is 2, not 358)
phi_error = np.where(phi_error > 180, 360 - phi_error, phi_error)

print(f"Mean Absolute MLAND Delta Phi: {np.mean(phi_error):.2f}°")
print(f"Standard Deviation of MLAND Delta Phi: {np.std(phi_error):.2f}°")

from scipy.optimize import curve_fit
import numpy as np
# 1. Define the Gaussian function
def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean)**2) / (2 * stddev**2))

from scipy.optimize import curve_fit

# Ensure weights_test is a 1D array to match the error arrays
# If weights_test was a DataFrame column or a (N, 1) array, flatten it.
weights = np.asarray(weights_test).flatten()

# --- Weighted Stats for Theta ---
w_mae_phi = np.average(phi_error, weights=weights)
w_var_phi = np.average((phi_error - w_mae_phi)**2, weights=weights)
w_std_phi = np.sqrt(w_var_phi)

print("-" * 30)
print(f"WEIGHTED STATS (using test weights):")
print(f"Weighted MAE Phi: {w_mae_phi:.2f}°")
print(f"Weighted Std Phi: {w_std_phi:.2f}°")
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
delta_phi_mland = (phi_pred_deg - phi_true_deg + 180) % 360 - 180

# ANITA Residuals (matching your logic from before)
# Note: Ensure the signs/math match your specific dataset convention here
delta_phi_anita = (df_test['phi_estimated'].to_numpy() - phi_true_deg.flatten() + 180) % 360 - 180

# --- ROW 0: STANDARD (Unweighted) ---
# Hexbin Comparison
hb0 = ax[0,0].hexbin(phi_true_deg, phi_pred_deg, gridsize=50, cmap='inferno', mincnt=1)
ax[0,0].plot([-180, 180], [-180, 180], 'w--', alpha=0.5)
ax[0,0].set_title("Phi Performance (Unweighted)")
ax[0,0].set_xlabel("True Phi (deg)")
ax[0,0].set_ylabel("Predicted Phi (deg)")
fig.colorbar(hb0, ax=ax[0,0], label='Counts')

# MLAND Gaussian Fit
plot_gaussian_fit(delta_phi_mland, ax[0,1], 
                  fit_range=(-2, 2), 
                  title="MLAND Residual (Unweighted)",
                  xlabel="$\Delta\\phi$ (deg)")

# ANITA Gaussian Fit
plot_gaussian_fit(delta_phi_anita, ax[0,2], 
                  fit_range=(-2, 2), 
                  title="ANITA Residual (Unweighted)",
                  xlabel="$\Delta\\phi$ (deg)")

# --- ROW 1: WEIGHTED ---
# Weighted Hexbin Comparison
# C=weights_array with np.sum means color = total weight in that bin
hb1 = ax[1,0].hexbin(phi_true_deg, phi_pred_deg, C=weights_array, reduce_C_function=np.sum, 
                     gridsize=50, cmap='viridis', mincnt=1)
ax[1,0].plot([-180, 180], [-180, 180], 'w--', alpha=0.5)
ax[1,0].set_title("Phi Performance (Weighted)")
ax[1,0].set_xlabel("True Phi (deg)")
ax[1,0].set_ylabel("Predicted Phi (deg)")
fig.colorbar(hb1, ax=ax[1,0], label='Sum of Weights')

# MLAND Weighted Gaussian Fit
plot_gaussian_fit(delta_phi_mland, ax[1,1], weights=weights_array, 
                  fit_range=(-2, 2), 
                  title="MLAND Residual (Weighted)",
                  xlabel="$\Delta\\phi$ (deg)")

# ANITA Weighted Gaussian Fit
plot_gaussian_fit(delta_phi_anita, ax[1,2], weights=weights_array, 
                  fit_range=(-2, 2), 
                  title="ANITA Residual (Weighted)",
                  xlabel="$\Delta\\phi$ (deg)")

#plt.tight_layout()
#plt.savefig(f"figs/int_phi_only_weighted_comparison_{tag}.png")

if False:
    plt.show()

# Final clean up
plt.tight_layout()
plt.savefig(f"figs/phi_performance_side_by_side_model_{tag}.png")
if False:
    plt.show()  

#
print("done and successfully saved the phi model and the predictions!")