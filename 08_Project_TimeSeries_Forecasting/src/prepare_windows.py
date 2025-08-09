"""
prepare_windows.py
------------------
Create train/val/test splits, scale (train-only), and make sliding windows.

Usage (single-series CSV with date,value):
  python -m src.prepare_windows --csv data/processed/series.csv --win 30 --horizon 1

Usage (from combined long file with multiple tickers):
  python -m src.prepare_windows --from-long data/processed/series_long.csv --ticker AAPL --win 30 --horizon 1

Outputs:
  data/processed/dataset.npz      # X_train, y_train, X_val, y_val, X_test, y_test, scaler stats, y_*_dates
  data/processed/dataset_meta.json
"""

import argparse
import os
import json
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def make_windows(series: np.ndarray,
                 dates: np.ndarray,
                 win: int,
                 horizon: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build sliding windows from a 1D scaled series.

    Returns:
      X: (n, win, 1)
      y: (n,)
      y_dates: (n,) pandas Timestamps aligned to the target y values (useful for plotting)
    """
    X, y, y_dates = [], [], []
    for i in range(len(series) - win - horizon + 1):
        X.append(series[i:i + win])
        y.append(series[i + win:i + win + horizon][0])  # univariate next-step target
        y_dates.append(dates[i + win + horizon - 1])
    X = np.asarray(X)[:, :, None]
    y = np.asarray(y)
    y_dates = np.asarray(y_dates)
    return X, y, y_dates


def load_series(args) -> pd.DataFrame:
    """
    Load a tidy series (date,value) either from:
      - --csv  (expects columns: date,value)
      - or from --from-long + --ticker (expects columns: date,ticker,value)
    """
    if args.csv and (args.from_long or args.ticker):
        raise ValueError("Use either --csv OR (--from-long AND --ticker), not both.")
    if args.csv:
        df = pd.read_csv(args.csv, parse_dates=["date"])
        if not {"date", "value"}.issubset(df.columns):
            raise ValueError(f"{args.csv} must have columns: date,value")
        return df[["date", "value"]].sort_values("date").reset_index(drop=True)

    if args.from_long and args.ticker:
        df_long = pd.read_csv(args.from_long, parse_dates=["date"])
        required = {"date", "ticker", "value"}
        if not required.issubset(df_long.columns):
            raise ValueError(f"{args.from_long} must have columns: {required}")
        sel = df_long["ticker"].astype(str).str.upper() == args.ticker.upper()
        df = df_long.loc[sel, ["date", "value"]].sort_values("date").reset_index(drop=True)
        if df.empty:
            raise ValueError(f"No rows found for ticker '{args.ticker}' in {args.from_long}")
        return df

    raise ValueError("Please provide either --csv OR (--from-long AND --ticker).")


def main():
    p = argparse.ArgumentParser()
    # Input options
    p.add_argument("--csv", default=None, help="Path to a single-series CSV with columns: date,value")
    p.add_argument("--from-long", default=None, help="Path to combined long CSV with columns: date,ticker,value")
    p.add_argument("--ticker", default=None, help="Ticker to extract from --from-long")
    # Windowing/splits
    p.add_argument("--win", type=int, default=30)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--test_ratio", type=float, default=0.15)
    # Output
    p.add_argument("--out_dir", default="data/processed")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load series (date,value)
    df = load_series(args)


    # NEW: force datetime (some CSVs load as strings)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df = df.dropna(subset=["date", "value"]).sort_values("date").reset_index(drop=True)

    values = df["value"].astype("float32").to_numpy().reshape(-1, 1)
    dates = df["date"].to_numpy()

    n = len(values)
    if n < args.win + args.horizon + 10:
        raise ValueError(f"Series too short (n={n}) for win={args.win} and horizon={args.horizon}.")

    n_test = int(n * args.test_ratio)
    n_val = int(n * args.val_ratio)
    n_train = n - n_val - n_test
    if min(n_train, n_val, n_test) <= args.win + args.horizon:
        # keep it friendly if splits too small
        raise ValueError(
            f"Splits too small for windowing. "
            f"Got train={n_train}, val={n_val}, test={n_test} with win={args.win}, horizon={args.horizon}."
        )

    train_vals, val_vals, test_vals = values[:n_train], values[n_train:n_train + n_val], values[n_train + n_val:]
    train_dates, val_dates, test_dates = dates[:n_train], dates[n_train:n_train + n_val], dates[n_train + n_val:]

    # Fit scaler on TRAIN only, transform all
    scaler = StandardScaler().fit(train_vals)
    train_s = scaler.transform(train_vals).ravel()
    val_s = scaler.transform(val_vals).ravel()
    test_s = scaler.transform(test_vals).ravel()

    # Windowing with carry-over to preserve continuity
    # Val windows see the last (win-1) points of train; Test windows see the last (win-1) of train+val
    val_input = np.concatenate([train_s[-(args.win - 1):], val_s]) if args.win > 1 else val_s
    val_input_dates = np.concatenate([train_dates[-(args.win - 1):], val_dates]) if args.win > 1 else val_dates

    tv_tail = np.concatenate([train_s[-(args.win - 1):], val_s]) if args.win > 1 else val_s
    tv_tail_dates = np.concatenate([train_dates[-(args.win - 1):], val_dates]) if args.win > 1 else val_dates
    test_input = np.concatenate([tv_tail[-(args.win - 1):], test_s]) if args.win > 1 else test_s
    test_input_dates = np.concatenate([tv_tail_dates[-(args.win - 1):], test_dates]) if args.win > 1 else test_dates

    Xtr, ytr, ytr_dates = make_windows(train_s, train_dates, args.win, args.horizon)
    Xva, yva, yva_dates = make_windows(val_input, val_input_dates, args.win, args.horizon)
    Xte, yte, yte_dates = make_windows(test_input, test_input_dates, args.win, args.horizon)

    # Save arrays + scaler stats + y dates (ISO format for easy plotting)
    out_npz = os.path.join(args.out_dir, "dataset.npz")
    np.savez(
        out_npz,
        X_train=Xtr, y_train=ytr,
        X_val=Xva,   y_val=yva,
        X_test=Xte,  y_test=yte,
        y_train_dates=ytr_dates.astype("datetime64[ns]"),
        y_val_dates=yva_dates.astype("datetime64[ns]"),
        y_test_dates=yte_dates.astype("datetime64[ns]"),
        scaler_mean=float(scaler.mean_[0]),
        scaler_scale=float(scaler.scale_[0])
    )

    # Save small meta JSON
    meta = {
        "n_total": int(n),
        "splits": {"train": int(n_train), "val": int(n_val), "test": int(n_test)},
        "win": int(args.win),
        "horizon": int(args.horizon),
        "source": args.csv if args.csv else args.from_long,
        "ticker": args.ticker,
        # NEW (safe even if strings sneak in)
        "date_min": pd.to_datetime(df["date"]).min().date().isoformat(),
        "date_max": pd.to_datetime(df["date"]).max().date().isoformat(),
    }
    with open(os.path.join(args.out_dir, "dataset_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved processed dataset to {out_npz}")
    print(f"Shapes: Xtr={Xtr.shape}  Xva={Xva.shape}  Xte={Xte.shape}")


if __name__ == "__main__":
    main()
