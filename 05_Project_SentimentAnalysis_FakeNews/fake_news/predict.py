#!/usr/bin/env python
"""
fake_news/predict.py
====================

• Scrapes a news-article URL (title + body) with newspaper3k
• Builds the 3-column feature frame that the calibrated TF-IDF pipeline
  expects:  text  |  text_length  |  domain
• Returns the probability the story is fake.

Usage (from project root):
    python -m fake_news.predict <url>
"""

import argparse, pathlib, joblib, tldextract, pandas as pd
from fake_news.fetch import scrape

PROJECT_DIR = pathlib.Path(__file__).resolve().parents[1]
MODEL_PATH  = PROJECT_DIR / "models" / "tfidf_logreg_calibrated.pkl"

if not MODEL_PATH.is_file():
    raise SystemExit("❌  Calibrated model not found -- run training first")

pipe = joblib.load(MODEL_PATH)


# ── helper to build the feature DataFrame ──────────────────────────────────
def make_feature_df(text: str, url: str) -> pd.DataFrame:
    df = pd.DataFrame({"text": [text]})
    df["text_length"] = len(text)
    domain = tldextract.extract(url).registered_domain or "unknown"
    df["domain"] = domain
    return df[["text", "text_length", "domain"]]


def predict(url: str) -> float:
    """Return P(fake) for the given article URL."""
    text = scrape(url)
    X    = make_feature_df(text, url)
    return float(pipe.predict_proba(X)[0, 1])


# ── CLI wrapper ────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("url", help="news-article URL to classify")
    args = ap.parse_args()

    p_fake = predict(args.url)
    verdict = "🟥 LIKELY FAKE" if p_fake > 0.5 else "🟩 LIKELY REAL"
    print(f"{verdict}  ({p_fake*100:.1f}% fake probability)")


if __name__ == "__main__":
    main()
