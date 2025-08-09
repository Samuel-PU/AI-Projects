#!/usr/bin/env python
"""
Interactive classifier: headline (+ optional body) → fake-probability.

• Loads the calibrated TF-IDF pipeline (models/tfidf_logreg_calibrated.pkl).
• Builds the same 3-column feature frame the model expects.
• If you leave the body blank it falls back to headline-only mode.

Run:
    python classify_text.py
"""

import pathlib, joblib, pandas as pd, tldextract

PROJECT_DIR = pathlib.Path(__file__).resolve().parents[1]
MODEL_PATH  = PROJECT_DIR / "fake-newsnet" / "models" / "tfidf_logreg_calibrated.pkl"

pipe = joblib.load(MODEL_PATH)


def make_df(text: str) -> pd.DataFrame:
    df = pd.DataFrame({"text": [text]})
    df["text_length"] = len(text)
    df["domain"] = "unknown"          # no URL for free-text
    return df[["text", "text_length", "domain"]]


def predict_free(text: str) -> float:
    X = make_df(text)
    return float(pipe.predict_proba(X)[0, 1])


def main() -> None:
    print("────────  Fake-News Mini Detector  ────────")
    headline = input("Paste headline: ").strip()
    if not headline:
        print("No headline entered. Exiting.")
        return

    body = input("Paste body text (or press Enter to skip): ").strip()
    full_text = f"{headline}\n{body}" if body else headline

    p = predict_free(full_text)
    verdict = "🟥 LIKELY FAKE" if p > 0.5 else "🟩 LIKELY REAL"
    print(f"\n{verdict}  ({p*100:.1f}% fake probability)")


if __name__ == "__main__":
    main()
