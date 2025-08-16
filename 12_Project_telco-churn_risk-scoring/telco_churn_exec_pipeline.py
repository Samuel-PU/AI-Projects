
# Telco Churn — Executive-Ready Classification Pipeline (Final + Calibration)
# --------------------------------------------------------------------------
# Sources: Kaggle (IBM Telco Customer Churn) + IBM docs; GitHub mirror for direct CSV.
# Kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
# IBM docs: https://www.ibm.com/docs/en/cognos-analytics/12.1.0?topic=samples-telco-customer-churn
# Mirror used for loading: https://raw.githubusercontent.com/Rizal-A/EDA-Telco_Customer_Churn/main/WA_Fn-UseC_-Telco-Customer-Churn.csv
#
# What’s new vs your draft:
# - Fixed threshold selection bug (precision/recall arrays vs thresholds length mismatch)
# - Added full calibration evaluation (reliability curve + CSV table)
# - Ensured models expose probabilities (SVM wrapped in CalibratedClassifierCV)
# - Cleaned preprocessing to dense outputs (OneHotEncoder(sparse=False)) for compatibility
# - Extra safety utils and improved report section (includes calibration artefacts)

import os, textwrap, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score, precision_recall_curve,
    brier_score_loss, confusion_matrix, classification_report, RocCurveDisplay,
    PrecisionRecallDisplay,
)
from sklearn.metrics import make_scorer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.inspection import permutation_importance
from datetime import datetime

# ---------------------- Config ----------------------
RANDOM_STATE = 42
MIN_PRECISION = 0.80
ARTIFACTS = Path("artifacts"); ARTIFACTS.mkdir(exist_ok=True)
DATA_URL = ("https://raw.githubusercontent.com/Rizal-A/EDA-Telco_Customer_Churn/"
            "main/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# ---------------------- Load & Clean ----------------------
df = pd.read_csv(DATA_URL)
# Standardise target
df["Churn"] = (df["Churn"].astype(str).str.strip().str.lower() == "yes").astype(int)
# Known quirk: TotalCharges has blanks; coerce to numeric and drop missing
if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)

# Unique key not needed for modelling
id_cols = [c for c in df.columns if c.lower() in {"customerid"}]
df = df.drop(columns=id_cols)

target = "Churn"
y = df[target].astype(int)
X = df.drop(columns=[target])

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

# ---------------------- Preprocess ----------------------
# Using dense output for reliability with all estimators on this small dataset.
num_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler())
])
cat_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))

])
pre = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
])

# ---------------------- Models ----------------------
models = {
    "dummy": DummyClassifier(strategy="most_frequent"),
    "logit": LogisticRegression(
        max_iter=2000, class_weight="balanced", solver="lbfgs", random_state=RANDOM_STATE
    ),
    "hgbt": HistGradientBoostingClassifier(
        learning_rate=0.1, max_depth=None, random_state=RANDOM_STATE
    ),
    # Use 'estimator=' (newer sklearn) instead of deprecated 'base_estimator='
    "svm_lin_cal": CalibratedClassifierCV(
        estimator=LinearSVC(C=1.0, class_weight="balanced", random_state=RANDOM_STATE),
        method="sigmoid", cv=3
    ),
}

# ---------------------- Custom Scorers ----------------------
def recall_at_precision_score(y_true, y_score, min_precision=MIN_PRECISION):
    # y_score can be probas or decision scores; precision_recall_curve handles either
    p, r, _ = precision_recall_curve(y_true, y_score)
    mask = p >= min_precision
    return float(r[mask].max()) if np.any(mask) else 0.0

recall_at_p80 = make_scorer(recall_at_precision_score, needs_proba=True)

scoring = {
    "pr_auc": "average_precision",
    "roc_auc": "roc_auc",
    "f1": "f1",
    "brier": make_scorer(brier_score_loss, needs_proba=True, greater_is_better=False),
    "recall_at_p80": recall_at_p80
}

# ---------------------- Split ----------------------
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# ---------------------- Cross-validate ----------------------
def evaluate_models():
    rows = []
    for name, clf in models.items():
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        cvres = cross_validate(
            pipe, X_tr, y_tr, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False
        )
        row = {"model": name}
        for key, vals in cvres.items():
            if key.startswith("test_"):
                metric = key.replace("test_", "")
                mean = float(np.mean(vals))
                # Flip brier to positive (lower is better) for readability
                if metric == "brier":
                    mean = -mean
                row[metric] = mean
        rows.append(row)
    return pd.DataFrame(rows).sort_values("pr_auc", ascending=False)

cv_table = evaluate_models()
cv_table.to_csv(ARTIFACTS / "cv_results.csv", index=False)
print("\nCross-validated metrics (mean over folds):\n")
print(cv_table)

best_name = str(cv_table.iloc[0]["model"])
print(f"\nSelected best model by PR-AUC: {best_name}")

best_pipe = Pipeline([("pre", pre), ("clf", models[best_name])]).fit(X_tr, y_tr)

# ---------------------- Threshold selection on validation-like split ----------------------
# IMPORTANT FIX: precision/recall arrays have len = n_thr+1; thresholds have len = n_thr.
# Align by slicing p[:-1], r[:-1] to index thresholds safely.
def pick_threshold_at_precision(y_true, y_score, min_precision=MIN_PRECISION, fallback=0.5):
    p, r, thr = precision_recall_curve(y_true, y_score)
    p_adj, r_adj = p[:-1], r[:-1]
    if p_adj.size == 0:
        return fallback, 0.0
    mask = p_adj >= min_precision
    if not np.any(mask):
        return fallback, 0.0
    # Among qualifying points, pick the one with highest recall
    best_idx = np.argmax(r_adj[mask])
    thr_star = float(thr[mask][best_idx])
    opt_recall = float(r_adj[mask][best_idx])
    return thr_star, opt_recall

# Internal split for threshold selection
X_tr2, X_val, y_tr2, y_val = train_test_split(
    X_tr, y_tr, test_size=0.20, stratify=y_tr, random_state=RANDOM_STATE
)
best_pipe.fit(X_tr2, y_tr2)

def get_scores(estimator, X):
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    # Fallback to decision scores if no proba (should not happen for defined models)
    if hasattr(estimator, "decision_function"):
        # Map to (0,1) via logistic for calibration-like scale
        z = estimator.decision_function(X)
        return 1.0 / (1.0 + np.exp(-z))
    # Last resort
    return estimator.predict(X)

proba_val = get_scores(best_pipe, X_val)
thr_star, opt_recall = pick_threshold_at_precision(y_val, proba_val, MIN_PRECISION, fallback=0.5)

print(f"\nChosen operating point: Precision ≥ {MIN_PRECISION:.2f} | "
      f"Recall={opt_recall:.3f} at threshold={thr_star:.3f}")

# ---------------------- Hold-out Test Evaluation ----------------------
proba_te = get_scores(best_pipe, X_te)
y_pred_te = (proba_te >= thr_star).astype(int)

metrics = {
    "PR_AUC": float(average_precision_score(y_te, proba_te)),
    "ROC_AUC": float(roc_auc_score(y_te, proba_te)),
    "F1_at_thr": float(f1_score(y_te, y_pred_te)),
    "Recall_at_thr": float((y_pred_te[y_te==1]==1).mean() if (y_te==1).any() else 0.0),
    "Precision_at_thr": float((y_te[y_pred_te==1]==1).mean()) if (y_pred_te==1).any() else 0.0,
    "Brier": float(brier_score_loss(y_te, proba_te)),
}
print("\nTest metrics at chosen threshold:\n", json.dumps(metrics, indent=2))

print("\nConfusion matrix (test @ threshold):")
print(confusion_matrix(y_te, y_pred_te))
print("\nClassification report (test @ threshold):")
print(classification_report(y_te, y_pred_te, digits=3))

# ---------------------- Plots ----------------------
def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

# PR & ROC
PrecisionRecallDisplay.from_predictions(y_te, proba_te)
plt.title("Precision-Recall (Test)")
savefig(ARTIFACTS / "pr_curve_test.png")

RocCurveDisplay.from_predictions(y_te, proba_te)
plt.title("ROC (Test)")
savefig(ARTIFACTS / "roc_curve_test.png")

# ---------------------- Calibration: Reliability Curve + CSV ----------------------
# Reliability diagram for the selected model on test
disp = CalibrationDisplay.from_predictions(y_te, proba_te, n_bins=10, strategy="quantile")
plt.title("Calibration (Reliability) — Test")
savefig(ARTIFACTS / "calibration_test.png")

# Export calibration table for auditing
def calibration_table(y_true, y_prob, n_bins=10, strategy="quantile"):
    # Bin by predicted probability
    if strategy == "quantile":
        bins = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
        # Deduplicate edges (can happen if proba ties)
        bins = np.unique(bins)
    else:
        bins = np.linspace(0, 1, n_bins + 1)
    # Assign bins
    idx = np.digitize(y_prob, bins, right=True) - 1
    # Fix edge cases
    idx = np.clip(idx, 0, len(bins) - 2)
    rows = []
    for b in range(len(bins) - 1):
        mask = idx == b
        if not np.any(mask):
            rows.append({
                "bin": b,
                "left": bins[b],
                "right": bins[b+1],
                "count": 0,
                "pred_mean": np.nan,
                "frac_pos": np.nan,
            })
            continue
        yp = y_prob[mask]
        yt = y_true[mask]
        rows.append({
            "bin": b,
            "left": float(bins[b]),
            "right": float(bins[b+1]),
            "count": int(mask.sum()),
            "pred_mean": float(np.mean(yp)),
            "frac_pos": float(np.mean(yt)),
        })
    return pd.DataFrame(rows)

cal_df = calibration_table(y_te.values, np.asarray(proba_te), n_bins=10, strategy="quantile")
cal_df.to_csv(ARTIFACTS / "calibration_table.csv", index=False)

# ---------------------- Permutation importance (PR-AUC as scoring) ----------------------
r = permutation_importance(best_pipe, X_te, y_te, n_repeats=10,
                           random_state=RANDOM_STATE, scoring="average_precision", n_jobs=-1)

# Build readable feature names
def feature_names_from_preprocessor(preprocessor, num_cols, cat_cols):
    names = []
    names.extend(list(num_cols))
    ohe = preprocessor.named_transformers_["cat"].named_steps["oh"]
    cat_names = ohe.get_feature_names_out(cat_cols)
    names.extend(list(cat_names))
    return names

feat_names = feature_names_from_preprocessor(best_pipe.named_steps["pre"], num_cols, cat_cols)
importances = pd.DataFrame({
    "feature": feat_names,
    "importance_mean": r.importances_mean,
    "importance_std": r.importances_std
}).sort_values("importance_mean", ascending=False).head(20)
importances.to_csv(ARTIFACTS / "top_features.csv", index=False)

plt.figure()
plt.barh(importances["feature"][::-1], importances["importance_mean"][::-1])
plt.xlabel("Permutation Importance (PR-AUC drop)")
plt.title("Top Drivers (Test)")
savefig(ARTIFACTS / "top_features.png")

# ---------------------- Mini Markdown Report ----------------------
pos_rate = float(y.mean())
now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
report_md = f"""# Telco Customer Churn — Executive Summary

**Run time:** {now}

## Objective
Predict customer churn to enable proactive retention. Primary metric: PR-AUC; operating point selected to satisfy Precision ≥ {MIN_PRECISION:.2f}.

## Data
Rows: {len(df):,} | Features: {X.shape[1]} | Positive rate: {pos_rate:.3f}

## Model Comparison (5-fold CV)
{cv_table.to_markdown(index=False)}

## Selected Model
**{best_name}** — best mean PR-AUC across folds.

## Test Metrics @ threshold={thr_star:.3f}
```json
{json.dumps(metrics, indent=2)}
```

## Calibration
- Reliability diagram: `calibration_test.png`
- Calibration table (quantile-binned): `calibration_table.csv`
- Brier score (lower is better): {metrics["Brier"]:.4f}

## Top Drivers
See `top_features.png` and `top_features.csv` (Permutation importance on test, scoring=PR-AUC).

## Plots
- `pr_curve_test.png` — Precision–Recall (test)
- `roc_curve_test.png` — ROC (test)
- `calibration_test.png` — Calibration (test)
- `top_features.png` — Top drivers

## Risks & Next Steps
- Monitor class balance & drift; recalibrate quarterly.
- Add tenure interactions and contract/payment features with monotonic constraints in GBDT.
- Fairness: compare group metrics by demographics; mitigate if gaps >5–10%.
"""
(ARTIFACTS / "report.md").write_text(report_md, encoding="utf-8")
print(f"\nArtifacts saved to: {ARTIFACTS.resolve()}")
