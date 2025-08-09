"""
backtest.py
-----------
Rolling-origin (walk-forward) evaluation on a single series.
Per fold, train LSTM/GRU on expanding window, validate on the next chunk,
then forecast the next step for the test point(s).

Usage:
  # from a combined long file
  python -m src.backtest --from-long data/processed/series_long.csv --ticker AAPL --model lstm --folds 6 --win 30

  # or from a single series
  python -m src.backtest --csv data/processed/series.csv --model gru --folds 6 --win 60
"""
import argparse, os, json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from .models import build_lstm, build_gru
from .utils import set_seed, set_plot_theme, rmse, mae, mape

def windowize(x, win):
    X, y = [], []
    for i in range(len(x) - win):
        X.append(x[i:i+win])
        y.append(x[i+win])
    X = np.asarray(X)[:, :, None]
    y = np.asarray(y)
    return X, y

def main():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", default=None, help="Single series CSV with date,value")
    src.add_argument("--from-long", default=None, help="Combined CSV date,ticker,value")
    p.add_argument("--ticker", default=None, help="Ticker to pick from --from-long")
    p.add_argument("--model", choices=["lstm","gru","none"], default="lstm",
                   help="If 'none', only baselines are computed (no NN training).")
    p.add_argument("--folds", type=int, default=6)
    p.add_argument("--win", type=int, default=30)
    p.add_argument("--units", type=int, default=64)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="output/backtest.json")
    p.add_argument("--fig", default="output/figures/backtest.png")
    args = p.parse_args()

    set_seed(args.seed)
    set_plot_theme()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(os.path.dirname(args.fig), exist_ok=True)

    # Load series
    if args.csv:
        df = pd.read_csv(args.csv, parse_dates=["date"]).sort_values("date")
    else:
        if not args.ticker:
            raise ValueError("Please provide --ticker when using --from-long.")
        df_long = pd.read_csv(args["from_long"] if isinstance(args, dict) else args.from_long, parse_dates=["date"])
        df = df_long[df_long["ticker"].astype(str).str.upper() == args.ticker.upper()][["date","value"]].sort_values("date")
        if df.empty:
            raise ValueError(f"No data for ticker {args.ticker}")

    values = df["value"].astype("float32").to_numpy()
    dates = df["date"].to_numpy()

    n = len(values)
    step = max((n - args.win) // (args.folds + 1), 1)

    records = []
    preds = []

    for k in range(args.folds):
        end = args.win + k*step
        if end + 1 >= n:  # need at least one step ahead
            break

        train = values[:end]
        val_end = min(end + step, n-1)
        val = values[end:val_end]  # small validation slice (not strictly used in 1-step forecast)
        # Train model if requested
        if args.model != "none":
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler().fit(train.reshape(-1,1))
            train_s = scaler.transform(train.reshape(-1,1)).ravel()
            Xtr, ytr = windowize(train_s, args.win)

            if args.model == "lstm":
                model = build_lstm(args.win, units=args.units)
            else:
                model = build_gru(args.win, units=args.units)
            model.compile(optimizer=Adam(args.lr), loss="mse")

            model.fit(
                Xtr, ytr,
                epochs=args.epochs,
                batch_size=args.batch,
                verbose=0,
                callbacks=[EarlyStopping(monitor="loss", patience=3, restore_best_weights=True)]
            )

            # 1-step ahead forecast at 'end' (using last window)
            last_win = scaler.transform(train[-args.win:].reshape(-1,1)).ravel().reshape(1, args.win, 1)
            yhat_1 = model.predict(last_win, verbose=0).ravel()[0]
            yhat_1 = yhat_1 * float(scaler.scale_[0]) + float(scaler.mean_[0])
        else:
            yhat_1 = np.nan  # no NN prediction

        # Ground truth at next time step
        y_true = values[end]  # the next point after last train index
        preds.append({"fold": k+1, "date": str(dates[end]), "y_true": float(y_true), "y_model": float(yhat_1)})

        # Track metric per fold if model exists
        if args.model != "none" and not np.isnan(yhat_1):
            records.append({
                "fold": k+1,
                "rmse": float(np.sqrt((y_true - yhat_1)**2)),
                "mae":  float(np.abs(y_true - yhat_1)),
                "mape": float(np.abs((y_true - yhat_1) / (y_true + 1e-8)) * 100.0),
            })

    # Save JSON summary
    out = {
        "model": args.model,
        "folds": len(preds),
        "predictions": preds,
        "summary": {
            "rmse_mean": float(np.mean([r["rmse"] for r in records])) if records else None,
            "mae_mean":  float(np.mean([r["mae"] for r in records])) if records else None,
            "mape_mean": float(np.mean([r["mape"] for r in records])) if records else None,
        }
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[OK] Backtest saved → {args.out}")

    # Plot: dots of 1-step predictions vs actual at each fold date
    dfp = pd.DataFrame(preds)
    if not dfp.empty:
        fig, ax = plt.subplots(figsize=(9,4))
        base = pd.DataFrame({"date": dates, "value": values})
        sns.lineplot(data=base, x="date", y="value", ax=ax, label="Series", alpha=0.6)
        sns.scatterplot(data=dfp, x="date", y="y_true", ax=ax, label="True @ fold", s=40)
        if args.model != "none":
            sns.scatterplot(data=dfp, x="date", y="y_model", ax=ax, label=f"{args.model.upper()} forecast", s=40)
        ax.set_title("Walk-Forward Backtest (1-step ahead)")
        ax.set_xlabel("Date"); ax.set_ylabel("Value")
        fig.tight_layout()
        fig.savefig(args.fig, dpi=180)
        plt.close(fig)
        print(f"[OK] Backtest plot     → {args.fig}")

if __name__ == "__main__":
    main()
