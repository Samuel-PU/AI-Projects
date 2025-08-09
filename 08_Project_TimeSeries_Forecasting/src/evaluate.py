# evaluate.py
import argparse, os, json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf

def inverse_scale(z, mean, scale):
    return z * float(scale) + float(mean)

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def moving_average(a, w):
    if w <= 1:
        return a.copy()
    out = np.full_like(a, np.nan, dtype=float)
    if w <= len(a):
        c = np.cumsum(np.insert(a, 0, 0.0))
        out[w-1:] = (c[w:] - c[:-w]) / w
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True, help="Path to saved .keras model")
    p.add_argument("--data", default="data/processed/dataset.npz", help="NPZ produced by prepare_windows.py")
    p.add_argument("--out",  default="output/plots", help="Directory to save plots and metrics")
    # optional baseline flags (computed on the test target)
    p.add_argument("--csv", default=None, help="(optional) ignored unless you need it; kept for CLI compatibility")
    p.add_argument("--season", type=int, default=None, help="Season length for seasonal naive baseline")
    p.add_argument("--ma", type=int, default=None, help="Window for moving-average baseline")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # --- Load dataset ---
    D = np.load(args.data)
    Xte, yte = D["X_test"], D["y_test"]
    mean = D["scaler_mean"].item() if hasattr(D["scaler_mean"], "item") else float(D["scaler_mean"])
    scale = D["scaler_scale"].item() if hasattr(D["scaler_scale"], "item") else float(D["scaler_scale"])

    # --- Load model WITHOUT compiling (avoids custom-metric deserialization) ---
    model = tf.keras.models.load_model(args.model_path, compile=False)

    # --- Predict (in scaled space), then invert scaling ---
    yhat_scaled = model.predict(Xte, verbose=0).ravel()
    y_true = inverse_scale(yte, mean, scale)
    y_pred = inverse_scale(yhat_scaled, mean, scale)

    # --- Metrics ---
    metrics = {
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "mse":  float(mean_squared_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
    }
    print(f"MAE:  {metrics['mae']:.4f}")
    print(f"MSE:  {metrics['mse']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")

    # --- Baselines computed on the test target itself (index-aligned) ---
    comp = pd.DataFrame({
        "t": np.arange(len(y_true), dtype=int),
        "Actual": y_true,
        "Model":  y_pred,
    })

    # naive (t-1)
    naive = np.full(len(y_true), np.nan, dtype=float)
    if len(y_true) > 1:
        naive[1:] = y_true[:-1]
        comp["Naive"] = naive

    # seasonal naive
    if args.season and args.season > 0:
        s = args.season
        snaive = np.full(len(y_true), np.nan, dtype=float)
        if s < len(y_true):
            snaive[s:] = y_true[:-s]
        comp[f"SNaive_{s}"] = snaive

    # moving average
    if args.ma and args.ma > 1:
        comp[f"MA_{args.ma}"] = moving_average(y_true, args.ma)

    # --- Plots ---
    # 1) Actual vs Predicted (+ baselines if present)
    fig, ax = plt.subplots(figsize=(12, 6))
    for col in comp.columns:
        if col == "t":
            continue
        sns.lineplot(data=comp, x="t", y=col, label=col, ax=ax)
    ax.set_title("Test — Actual vs Predictions (and baselines)")
    ax.set_xlabel("Index (test timeline)")
    ax.set_ylabel("Value")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "pred_vs_actual.png"), dpi=180)
    plt.close(fig)

    # 2) Residual distribution (Model only)
    resid = comp["Actual"] - comp["Model"]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(resid.dropna(), kde=True, ax=ax)
    ax.set_title("Residual Distribution (Model)")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "residual_distribution.png"), dpi=180)
    plt.close(fig)

    # 3) Save metrics
    with open(os.path.join(args.out, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()
