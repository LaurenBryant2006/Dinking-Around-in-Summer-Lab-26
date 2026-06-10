## CNN model for ANITA neutrino direction reconstruction
## Inputs: raw_waveforms shape (48, 100, 2)
## Targets: mctheta, sin_phi, cos_phi

import polars as pl
import numpy as np
import gc
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


def get_X(df):
    return df["raw_waveforms"].to_numpy().astype(np.float32)


def gaussian(x, amplitude, mu, sigma):
    return amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def build_cnn():
    model = models.Sequential([
        layers.Input(shape=(48, 100, 2)),
        layers.Conv2D(32, kernel_size=(3, 5), activation="relu", padding="same"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 5), activation="relu", padding="same"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(3),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
    )
    return model


def main():
    overall_t0 = time.time()
    tf.keras.utils.set_random_seed(30)

    # ----------------------------------------------------------------
    # Load data
    base = "/fs/ess/PAS2159/HughesLab2/ANITA_DATA/parquet_for_fitting/"
    t0 = time.time()
    df_training = pl.read_parquet(base + "processed_training_data.parquet")
    df_test     = pl.read_parquet(base + "processed_testing_data.parquet")
    print(f"loaded data in {time.time() - t0:.1f}s")
    print("training:", df_training.shape)
    print("testing :", df_test.shape)

    # ----------------------------------------------------------------
    # Build X, y, weights
    X_train = get_X(df_training)
    X_test  = get_X(df_test)
    y_train = df_training[["mctheta", "sin_phi", "cos_phi"]].to_numpy().astype(np.float32)
    y_test  = df_test    [["mctheta", "sin_phi", "cos_phi"]].to_numpy().astype(np.float32)
    w_train = df_training["weight"].to_numpy().astype(np.float32)
    w_test  = df_test    ["weight"].to_numpy().astype(np.float32)

    phi_test_true   = df_test["mcphi"].to_numpy()
    theta_test_true = df_test["mctheta"].to_numpy()

    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_test :", X_test.shape,  "y_test :", y_test.shape)

    # Free dataframes — we've pulled what we need
    del df_training, df_test
    gc.collect()

    # ----------------------------------------------------------------
    # Normalize inputs (in-place, to avoid peak memory blowup)
    X_mean = X_train.mean(axis=(0, 1, 2), keepdims=True).astype(np.float32)
    X_std  = X_train.std (axis=(0, 1, 2), keepdims=True).astype(np.float32) + 1e-8

    X_train -= X_mean
    X_train /= X_std
    X_test  -= X_mean
    X_test  /= X_std
    print("after normalization, X_train mean:", X_train.mean(), "std:", X_train.std())

    # ----------------------------------------------------------------
    # Build and inspect model
    model = build_cnn()
    model.summary()

    # ----------------------------------------------------------------
    # Train
    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        sample_weight=w_train,
        validation_split=0.1,
        epochs=20,
        batch_size=64,
        callbacks=[
            callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            callbacks.ModelCheckpoint("cnn_best.keras", save_best_only=True, monitor="val_loss"),
        ],
        verbose=2,   # one line per epoch, instead of per-batch progress bars
    )
    print(f"trained in {time.time() - t0:.1f}s")
    model.save("cnn_final.keras")

    # ----------------------------------------------------------------
    # Loss curves
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"],     label="training loss")
    plt.plot(history.history["val_loss"], label="validation loss")
    plt.xlabel("epoch")
    plt.ylabel("loss (MSE)")
    plt.yscale("log")
    plt.legend()
    plt.title("Training history")
    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150)
    plt.close()

    # ----------------------------------------------------------------
    # Predict and compute errors
    pred = model.predict(X_test, batch_size=64, verbose=0)

    theta_pred = pred[:, 0]
    sin_pred   = pred[:, 1]
    cos_pred   = pred[:, 2]
    phi_pred   = np.degrees(np.arctan2(sin_pred, cos_pred)) % 360

    theta_err = theta_pred - theta_test_true
    phi_err   = (phi_pred - phi_test_true + 180) % 360 - 180

    print("UNWEIGHTED")
    print(f"  theta: mean={theta_err.mean():7.3f}   std={theta_err.std():7.3f}")
    print(f"  phi  : mean={phi_err.mean():7.3f}   std={phi_err.std():7.3f}")

    print("WEIGHTED")
    for name, err in [("theta", theta_err), ("phi", phi_err)]:
        mean_w = np.average(err, weights=w_test)
        var_w  = np.average((err - mean_w)**2, weights=w_test)
        std_w  = np.sqrt(var_w)
        print(f"  {name:5s}: mean={mean_w:7.3f}   std={std_w:7.3f}")

    # ----------------------------------------------------------------
    # Predicted vs true scatter
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(theta_test_true, theta_pred, s=2, alpha=0.3)
    axes[0].plot([6, 15], [6, 15], "r--")
    axes[0].set_xlim(6, 15); axes[0].set_ylim(6, 15); axes[0].set_aspect("equal")
    axes[0].set_xlabel("true theta"); axes[0].set_ylabel("predicted theta"); axes[0].set_title("theta")

    axes[1].scatter(phi_test_true, phi_pred, s=2, alpha=0.3)
    axes[1].plot([0, 360], [0, 360], "r--")
    axes[1].set_xlim(0, 360); axes[1].set_ylim(0, 360); axes[1].set_aspect("equal")
    axes[1].set_xlabel("true phi"); axes[1].set_ylabel("predicted phi"); axes[1].set_title("phi")

    plt.tight_layout()
    plt.savefig("pred_vs_true.png", dpi=150)
    plt.close()

    # ----------------------------------------------------------------
    # Delta plots with Gaussian fits
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, err, name, symbol in [
        (axes[0], theta_err, "theta", r"\Delta\theta"),
        (axes[1], phi_err,   "phi",   r"\Delta\phi"),
    ]:
        n_bins = 80
        counts, edges, _ = ax.hist(
            err, bins=n_bins, weights=w_test,
            edgecolor="black", linewidth=0.3,
        )
        centers = 0.5 * (edges[:-1] + edges[1:])

        mu0 = np.average(err, weights=w_test)
        sig0 = np.sqrt(np.average((err - mu0)**2, weights=w_test))
        p0 = [counts.max(), mu0, sig0]
        popt, _ = curve_fit(gaussian, centers, counts, p0=p0)
        amp_fit, mu_fit, sigma_fit = popt
        sigma_fit = abs(sigma_fit)

        x_smooth = np.linspace(err.min(), err.max(), 400)
        ax.plot(x_smooth, gaussian(x_smooth, *popt), "r-", linewidth=1.5,
                label=fr"fit: $\mu={mu_fit:.2f},\ \sigma={sigma_fit:.2f}$")

        ax.axvline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel(fr"${symbol}$ = predicted − true   (degrees)")
        ax.set_ylabel("weighted count")
        ax.set_title(name)
        ax.legend()

    plt.tight_layout()
    plt.savefig("delta_plots.png", dpi=150)
    plt.close()

    # ----------------------------------------------------------------
    # Total runtime
    elapsed = time.time() - overall_t0
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\ntotal wall time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print("all done")

if __name__ == "__main__":
    main()