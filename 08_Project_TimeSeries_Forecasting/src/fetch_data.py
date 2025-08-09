"""
fetch_data.py  (final)
----------------------
Load multiple raw OHLCV CSVs from data/raw/, clean each, and write:
  - data/processed/series_long.csv        (date,ticker,value)
  - data/processed/series_<TICKER>.csv    (optional per-ticker if --out-per-ticker)
  - data/processed/series_meta.json       (provenance/QA)

Examples:
  python -m src.fetch_data --tickers AAPL MSFT META
  python -m src.fetch_data --tickers aapl Meta MSFT_1986-03-13_2025-07-14 --use-adj-close --start 2015-01-01 --end 2025-01-01 --freq B --ffill-limit 2 --out-per-ticker
"""

import argparse
import os
import json
from typing import List, Dict

import numpy as np
import pandas as pd

# ----- Config -----
DATE_CANDIDATES = ["date", "datetime", "timestamp", "time", "period", "tradedate", "trade_date"]
PRICE_CANDIDATES_CLOSE_FIRST = ["close", "adj close", "adj_close", "adjclose", "close*"]  # columns normalised to lower-case


# ----- Low-level helpers -----
def _auto_read(csv_path: str) -> pd.DataFrame:
    """Read CSV with delimiter sniffing (handles ',', ';', tabs, etc.)."""
    return pd.read_csv(csv_path, engine="python", sep=None)


def _normalise_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names: strip, lowercase, collapse spaces."""
    new = []
    for c in df.columns:
        c2 = " ".join(str(c).strip().split()).lower()
        new.append(c2)
    df.columns = new
    return df


def _find_date_col(df: pd.DataFrame) -> str:
    cols = set(df.columns)
    # exact candidates
    for cand in DATE_CANDIDATES + ["date"]:
        if cand in cols:
            return cand
    # fuzzy contains
    for c in df.columns:
        if "date" in c or "time" in c:
            return c
    raise ValueError(f"Couldn’t find a date-like column. Available columns: {list(df.columns)}")


def _find_price_col(df: pd.DataFrame, use_adj_close: bool) -> str:
    cols = set(df.columns)
    if use_adj_close:
        for cand in ["adj close", "adj_close", "adjclose"]:
            if cand in cols:
                return cand
        # fall back to close if adj not present
    for cand in PRICE_CANDIDATES_CLOSE_FIRST + ["close"]:
        if cand in cols:
            return cand
    # last attempt: any col containing 'close'
    for c in df.columns:
        if "close" in c:
            return c
    raise ValueError(f"No close/adj close column found. Available columns: {list(df.columns)}")


def _to_datetime_utc_then_naive(s: pd.Series) -> pd.Series:
    """
    Parse to timezone-aware UTC, then strip tz to get tz-naive.
    Handles mixed offsets; avoids 'different timezones' errors later.
    """
    try:
        dt = pd.to_datetime(s, errors="coerce", utc=True)
    except Exception:
        # try numeric epochs
        s2 = pd.to_numeric(s, errors="coerce")
        if s2.notna().any():
            if s2.median() > 1e12:
                dt = pd.to_datetime(s2, unit="ms", errors="coerce", utc=True)
            else:
                dt = pd.to_datetime(s2, unit="s", errors="coerce", utc=True)
        else:
            dt = pd.to_datetime(s, errors="coerce", utc=True)
    # strip timezone -> tz-naive
    return dt.dt.tz_localize(None)


# ----- One-file cleaner -----
def clean_one(
    csv_path: str,
    ticker: str,
    use_adj_close: bool,
    start: str,
    end: str,
    freq: str,
    ffill_limit: int,
) -> pd.DataFrame:
    # 1) read with delimiter sniffing + normalise headers
    df = _auto_read(csv_path)
    df = _normalise_cols(df)

    # if weird headerless file: retry without header
    if len(df.columns) == 1 and df.columns[0].startswith("unnamed"):
        df = pd.read_csv(csv_path, header=None, engine="python", sep=None)
        df = _normalise_cols(df)

    # 2) identify columns
    date_col = _find_date_col(df)
    price_col = _find_price_col(df, use_adj_close)

    # 3) select, parse dates, de-dup, sort
    df = df[[date_col, price_col]].rename(columns={date_col: "date", price_col: "value"})
    df["date"] = _to_datetime_utc_then_naive(df["date"])
    df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")

    # 4) optional date window
    if start:
        df = df[df["date"] >= pd.to_datetime(start)]
    if end:
        df = df[df["date"] <= pd.to_datetime(end)]

    # 5) optional frequency reindex (e.g., business days)
    if freq and len(df):
        df = df.set_index("date").sort_index()
        full_idx = pd.date_range(df.index.min(), df.index.max(), freq=freq)
        df = df.reindex(full_idx)
        df["value"] = pd.to_numeric(df["value"], errors="coerce").ffill(limit=ffill_limit)
        # repair leading NaN if any
        if pd.isna(df["value"].iloc[0]) and len(df) > 1:
            df.iloc[0, df.columns.get_loc("value")] = df["value"].iloc[1]
        df.index.name = "date"
        df = df.reset_index()

    # 6) numeric cast & final dropna
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])

    df["ticker"] = ticker
    return df[["date", "ticker", "value"]]


# ----- CLI -----
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", nargs="+", required=True,
                   help="List of CSV base names in data/raw/ (without .csv), e.g. AAPL MSFT META")
    p.add_argument("--in-dir", default="data/raw", help="Folder containing raw CSVs")
    p.add_argument("--out-long", default="data/processed/series_long.csv", help="Combined tidy output path")
    p.add_argument("--out-per-ticker", action="store_true", help="Also write data/processed/series_<TICKER>.csv")
    p.add_argument("--meta-file", default="data/processed/series_meta.json", help="Provenance/QA output")
    p.add_argument("--use-adj-close", action="store_true", help="Prefer 'Adj Close' over 'Close' if available")
    p.add_argument("--start", default=None, help="YYYY-MM-DD (inclusive)")
    p.add_argument("--end", default=None, help="YYYY-MM-DD (inclusive)")
    p.add_argument("--freq", default=None, help="Reindex frequency, e.g. 'B' for business days")
    p.add_argument("--ffill-limit", type=int, default=2, help="Max consecutive forward-fills when reindexing")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out_long), exist_ok=True)

    all_frames: List[pd.DataFrame] = []
    meta: Dict[str, dict] = {"files": {}, "params": {
        "use_adj_close": bool(args.use_adj_close),
        "start": args.start, "end": args.end,
        "freq": args.freq, "ffill_limit": int(args.ffill_limit)
    }}

    for t in args.tickers:
        raw_path = os.path.join(args.in_dir, f"{t}.csv")
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"Missing raw file: {raw_path}")

        try:
            df_t = clean_one(raw_path, t, args.use_adj_close, args.start, args.end, args.freq, args.ffill_limit)
        except Exception as e:
            raise RuntimeError(f"Failed to process {t}: {e}")

        all_frames.append(df_t)
        meta["files"][t] = {
            "source_file": raw_path,
            "rows_out": int(len(df_t)),
            "date_min": df_t["date"].min().strftime("%Y-%m-%d") if len(df_t) else None,
            "date_max": df_t["date"].max().strftime("%Y-%m-%d") if len(df_t) else None,
            "used_col": "Adj Close" if args.use_adj_close else "Close/AdjClose (auto)"
        }

        if args.out_per_ticker:
            per_path = os.path.join(os.path.dirname(args.out_long), f"series_{t}.csv")
            df_t.to_csv(per_path, index=False)

    if not all_frames:
        raise RuntimeError("No data produced.")
    combined = pd.concat(all_frames, axis=0, ignore_index=True).sort_values(["ticker", "date"])
    combined.to_csv(args.out_long, index=False)

    with open(args.meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] Combined tidy series → {args.out_long} (rows={len(combined)})")
    print(f"[OK] Metadata             → {args.meta_file}")
    if args.out_per_ticker:
        print("[OK] Per-ticker CSVs      → data/processed/series_<TICKER>.csv")


if __name__ == "__main__":
    main()
