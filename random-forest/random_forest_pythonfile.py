## python file for code for a full random tree run

## necessary imports and  functions
import polars as pl
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit

def get_X(df):
    X = df["processed_waveforms"].to_numpy()
    if X.ndim ==1:
        X = np.stack(df["processed_waveforms"].to_list())
    return X

def gaussian(x, amplitude, mu, sigma):
    return amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def main():
#----------------------------------------------------------------------------------------------------------------------

## pulling from test and training data
    base = "/fs/ess/PAS2159/HughesLab2/ANITA_DATA/parquet_for_fitting/"
    df_training = pl.read_parquet( base + "processed_training_data.parquet")
    df_test = pl.read_parquet( base + "processed_testing_data.parquet")

    print("training:", df_training.shape)
    print("testing:", df_test.shape)
#----------------------------------------------------------------------------------------------------------------------

## creating inputs for random trees
    X_training = get_X(df_training)
    X_test = get_X(df_test)
    y_training = df_training[["mctheta", "sin_phi", "cos_phi"]].to_numpy() ## switched mcphi to sin/cos
    y_test = df_test[["mctheta", "sin_phi", "cos_phi"]].to_numpy()
    phi_test_true = df_test["mcphi"].to_numpy()
    theta_test_true = df_test["mctheta"].to_numpy()

    print("X_training", X_training.shape, "y_training", y_training.shape)
    print("X_test", X_test.shape, "y_test", y_test.shape)

#----------------------------------------------------------------------------------------------------------------------

## random forest! learning 
    w_train = df_training["weight"].to_numpy()
    print("about to train on:", X_training.shape,
          "y:", y_training.shape,
          "weights:", w_train.shape)
    rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=30)  
    rf.fit(X_training, y_training, sample_weight=w_train)
    print("done training")

    joblib.dump(rf, "rf_full_set_sincos.joblib") 
    print("saved")
#----------------------------------------------------------------------------------------------------------------------

## Error Calculations
    pred = rf.predict(X_test)
    w_test = df_test["weight"].to_numpy()
    print("pred:", pred.shape, "y_test:", y_test.shape, "w_test:", w_test.shape)

    theta_pred = pred[:, 0]
    theta_err  = theta_pred - theta_true_test

    sin_pred = pred[:, 1]
    cos_pred = pred[:, 2]
    phi_pred = np.degrees(np.arctan2(sin_pred, cos_pred)) % 360
    phi_err = (phi_pred - phi_test_true + 180) % 360 - 180

    print("UNWEIGHTED")
    print(f"  theta: mean={theta_err.mean():7.3f}   std={theta_err.std():7.3f}")
    print(f"  phi  : mean={phi_err.mean():7.3f}   std={phi_err.std():7.3f}")

    print("WEIGHTED")
    for name, err in [("theta", theta_err), ("phi", phi_err)]:
        mean_w = np.average(err, weights=w_test)
        var_w  = np.average((err - mean_w)**2, weights=w_test)
        std_w  = np.sqrt(var_w)
        print(f"  {name:5s}: mean={mean_w:7.3f}   std={std_w:7.3f}")
#----------------------------------------------------------------------------------------------------------------------

## PRED V TRUE Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(y_test[:, 0], theta_pred, s=2, alpha=0.3)
    axes[0].plot([6, 15], [6, 15], "r--")
    axes[0].set_xlim(6, 15); axes[0].set_ylim(6, 15); axes[0].set_aspect("equal")
    axes[0].set_xlabel("true theta"); axes[0].set_ylabel("predicted theta"); axes[0].set_title("theta")

    axes[1].scatter(phi_test_true, phi_pred, s=2, alpha=0.3)
    axes[1].plot([0, 360], [0, 360], "r--")
    axes[1].set_xlim(0, 360); axes[1].set_ylim(0, 360); axes[1].set_aspect("equal")
    axes[1].set_xlabel("true phi"); axes[1].set_ylabel("predicted phi"); axes[1].set_title("phi")

    plt.tight_layout()
    plt.savefig("pred_vs_true.png", dpi = 150)
    plt.close()
#----------------------------------------------------------------------------------------------------------------------

## Delta Plots
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

        p0 = [counts.max(),
              np.average(err, weights=w_test),
              np.sqrt(np.average((err - np.average(err, weights=w_test)) ** 2, weights=w_test))]
        popt, pcov = curve_fit(gaussian, centers, counts, p0=p0)
        amp_fit, mu_fit, sigma_fit = popt
        sigma_fit = abs(sigma_fit)   # curve_fit may return negative sigma; only magnitude is meaningful

        x_smooth = np.linspace(err.min(), err.max(), 400)
        ax.plot(x_smooth, gaussian(x_smooth, *popt), "r-", linewidth=1.5,
                label=fr"fit: $\mu={mu_fit:.2f},\ \sigma={sigma_fit:.2f}$")

        ax.axvline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel(fr"${symbol}$ = predicted − true   (degrees)")
        ax.set_ylabel("weighted count")
        ax.set_title(name)
        ax.legend()

    plt.tight_layout()
    plt.savefig("delta_plots.png", dpi = 150)
    plt.close()

if __name__ == "__main__":
    main()