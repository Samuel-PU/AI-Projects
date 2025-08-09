#!/usr/bin/env python
"""fake_news/plot_metrics.py

✔ Works with *either* model format:
   • Calibrated Pipeline (`tfidf_logreg_calibrated.pkl`)
   • Legacy dict bundle (`tfidf_logreg.pkl`)

Automatically builds the minimal DataFrame columns expected by the
Pipeline (text, text_length, domain) so `FunctionTransformer`s don’t
raise KeyError.
"""

import argparse, pathlib, sys, joblib, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import tldextract

# very top of plot_metrics.py – BEFORE 'import matplotlib.pyplot as plt'
import matplotlib
matplotlib.use("Agg")        # use non-GUI backend
import matplotlib.pyplot as plt
...



PROJECT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))  # ensure package importable

CSV_PATH   = PROJECT_DIR / "data" / "processed" / "train.csv"
CAL_MODEL  = PROJECT_DIR / "models" / "tfidf_logreg_calibrated.pkl"
LEG_MODEL  = PROJECT_DIR / "models" / "tfidf_logreg.pkl"
MODEL_PATH = CAL_MODEL if CAL_MODEL.is_file() else LEG_MODEL


def make_feature_df(text_series: pd.Series, url_series: pd.Series | None):
    """Return DataFrame with columns the Pipeline expects."""
    df = pd.DataFrame({"text": text_series})
    df["text_length"] = text_series.str.len()
    if url_series is not None:
        df["domain"] = url_series.fillna("").apply(
            lambda u: tldextract.extract(u).registered_domain or "unknown")
    else:
        df["domain"] = "unknown"
    return df[["text", "text_length", "domain"]]


def load_proba(model_obj, X_df, X_text):
    """Return P(fake) using appropriate input shape."""
    if hasattr(model_obj, "predict_proba"):
        return model_obj.predict_proba(X_df)[:, 1]
    # legacy dict bundle
    tfidf, clf = model_obj["tfidf"], model_obj["clf"]
    return clf.predict_proba(tfidf.transform(X_text))[:, 1]


def main(test_size: float = 0.2):
    if not CSV_PATH.is_file() or not MODEL_PATH.is_file():
        raise SystemExit("CSV or model file missing – run training first.")

    raw = pd.read_csv(CSV_PATH)
    url_col = next((c for c in raw.columns if c.lower().endswith("url")), None)
    X_train, X_test, y_train, y_test = train_test_split(
        raw, raw["label"], test_size=test_size, stratify=raw["label"], random_state=42
    )

    X_test_full = make_feature_df(X_test["text"], X_test[url_col] if url_col else None)

    model_obj = joblib.load(MODEL_PATH)
    proba = load_proba(model_obj, X_test_full, X_test["text"])
    preds = (proba > 0.5).astype(int)

    print("\n" + classification_report(y_test, preds, target_names=["real", "fake"], digits=3))
    roc_auc = auc(*roc_curve(y_test, proba)[:2])
    print(f"ROC-AUC: {roc_auc:.3f}\n")

    cm = confusion_matrix(y_test, preds)
    fig_cm, ax_cm = plt.subplots(figsize=(4, 4))
    ax_cm.imshow(cm, cmap="Blues")
    ax_cm.set_xticks([0, 1], ["real", "fake"])
    ax_cm.set_yticks([0, 1], ["real", "fake"])
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("True")
    for i in range(2):
        for j in range(2):
            ax_cm.text(j, i, cm[i, j], ha="center", va="center")
    ax_cm.set_title("Confusion Matrix")
    fig_cm.tight_layout()

    fpr, tpr, _ = roc_curve(y_test, proba)
    fig_roc, ax_roc = plt.subplots(figsize=(4.5, 4.5))
    ax_roc.plot(fpr, tpr, linewidth=2)
    ax_roc.plot([0, 1], [0, 1], linestyle="--")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title(f"ROC Curve (AUC = {roc_auc:.3f})")
    fig_roc.tight_layout()

    plt.show()
    # replace final plt.show()
    fig_cm.savefig("confusion.png", dpi=120, bbox_inches="tight")
    fig_roc.savefig("roc_curve.png", dpi=120, bbox_inches="tight")
    print("🔸 Charts saved: confusion.png  roc_curve.png")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--test_size", type=float, default=0.2)
    args = p.parse_args()
    main(test_size=args.test_size)
