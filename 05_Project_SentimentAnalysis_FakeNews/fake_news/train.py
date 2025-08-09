#!/usr/bin/env python
"""
Enhanced TF-IDF + LogisticRegression trainer with:

• word + char n-grams, text-length, and (optional) domain one-hot
• 5-fold stratified ROC-AUC estimate
• probability calibration (sigmoid)
• class-balanced liblinear solver
• saves to models/tfidf_logreg_calibrated.pkl
"""

import argparse, pathlib, joblib, pandas as pd, tldextract
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.calibration import CalibratedClassifierCV

# helper functions now imported so they pickle cleanly
from fake_news.helpers import col_text, col_len, col_domain

PROJECT_DIR  = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_CSV  = PROJECT_DIR / "data" / "processed" / "train.csv"
MODEL_PATH   = PROJECT_DIR / "models" / "tfidf_logreg_calibrated.pkl"
MODEL_PATH.parent.mkdir(exist_ok=True)

# ──────────────────────────────────────────────────────────────
def build_pipeline(include_domain: bool = True):
    word_vec = TfidfVectorizer(max_features=20_000,
                               ngram_range=(1, 2),
                               stop_words="english",
                               min_df=2)
    char_vec = TfidfVectorizer(analyzer="char",
                               ngram_range=(3, 5),
                               max_features=7_000,
                               min_df=3)
    text_pipe = FeatureUnion([("w", word_vec), ("c", char_vec)])

    parts = [
        ("text",   make_pipeline(FunctionTransformer(col_text,  validate=False), text_pipe)),
        ("length", make_pipeline(FunctionTransformer(col_len,   validate=False)))
    ]
    if include_domain:
        parts.append(
            ("domain",
             make_pipeline(
                 FunctionTransformer(col_domain, validate=False),
                 OneHotEncoder(handle_unknown="ignore" , sparse=True)))
        )
    preproc = FeatureUnion(parts)

    base_clf = LogisticRegression(solver="liblinear",
                                  class_weight="balanced",
                                  max_iter=2000,
                                  C=0.5,
                                  random_state=42)
    return make_pipeline(preproc, base_clf)
# ──────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV, type=pathlib.Path)
    args = ap.parse_args()

    if not args.csv.is_file():
        raise SystemExit(f"CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)
    df["text_length"] = df["text"].str.len()

    # domain extraction if URL present
    url_col = next((c for c in df.columns if c.lower().endswith("url")), None)
    if url_col:
        df["domain"] = df[url_col].fillna("").apply(
            lambda u: tldextract.extract(u).registered_domain or "unknown")
        use_domain = True
    else:
        df["domain"] = "unknown"
        use_domain = False

    X = df[["text", "text_length", "domain"]]
    y = df["label"].values

    pipe = build_pipeline(use_domain)

    # 5-fold CV AUC
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"\n📈 5-fold ROC-AUC: {aucs.mean():.3f} ± {aucs.std():.3f}")

    # probability-calibrated model
    calib = CalibratedClassifierCV(pipe, cv=5, method="sigmoid", n_jobs=-1)

    # hold-out report
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    calib.fit(X_tr, y_tr)
    proba = calib.predict_proba(X_te)[:, 1]
    preds = (proba > 0.5)

    print("\nClassification report (20 % hold-out):")
    print(classification_report(y_te, preds, digits=3))
    print("Hold-out ROC-AUC:", round(roc_auc_score(y_te, proba), 3))

    joblib.dump(calib, MODEL_PATH)
    print(f"\n✅  Saved calibrated model → {MODEL_PATH.relative_to(PROJECT_DIR)}")

if __name__ == "__main__":
    main()
