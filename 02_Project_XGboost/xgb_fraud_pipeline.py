#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve, roc_curve,
    confusion_matrix, f1_score, precision_score, recall_score
)
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
import xgboost as xgb

# Optional: SHAP for interpretation (safe to skip if not installed)
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


# -------------------- Utils --------------------
def parse_args():
    p = argparse.ArgumentParser(description="XGBoost fraud detection with advanced evaluation & plots.")
    p.add_argument("--data", type=str, default=None, help="Path to CSV data file (optional).")
    p.add_argument("--target", type=str, default=None, help="Target column (e.g., Class/Approved). If omitted, will try to infer.")
    p.add_argument("--out", type=str, default="outputs", help="Output directory.")
    p.add_argument("--test-size", type=float, default=0.2, help="Test split size.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")

    # XGB hyperparameters
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--n-estimators", type=int, default=600)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--subsample", type=float, default=0.9)
    p.add_argument("--colsample-bytree", type=float, default=0.9)
    p.add_argument("--min-child-weight", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=0.0)
    p.add_argument("--reg-alpha", type=float, default=0.0)
    p.add_argument("--reg-lambda", type=float, default=1.0)

    p.add_argument("--calibration-bins", type=int, default=10, help="Bins for calibration curve.")
    return p.parse_args()


def compute_scale_pos_weight(y):
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    return 1.0 if pos == 0 else float(neg) / float(pos)


def to_binary(series):
    """Map common label encodings (Y/N, +/- etc.) to {0,1}."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int).values
    s = series.astype(str).str.strip().str.lower()
    pos_tokens = {"1", "true", "t", "yes", "y", "approved", "+", "pos", "positive"}
    neg_tokens = {"0", "false", "f", "no", "n", "denied", "-", "neg", "negative"}
    out = []
    for v in s:
        if v in pos_tokens:
            out.append(1)
        elif v in neg_tokens:
            out.append(0)
        else:
            out.append(1 if "approve" in v else 0)
    return np.array(out, dtype=int)


def choose_best_threshold(y_true, y_score, beta=1.0):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    thresholds = np.append(thresholds, 1.0)  # align lengths
    beta2 = beta * beta
    f_scores = (1 + beta2) * (precision * recall) / np.maximum((beta2 * precision + recall), 1e-12)
    best_idx = int(np.nanargmax(f_scores))
    return float(thresholds[best_idx]), float(f_scores[best_idx]), float(precision[best_idx]), float(recall[best_idx])


def predict_proba_best(model, X):
    """Predict using the best iteration if supported across XGBoost versions."""
    # Newer API (>=1.6): use iteration_range with best_iteration
    try:
        return model.predict_proba(X, iteration_range=(0, model.best_iteration + 1))[:, 1]
    except Exception:
        pass
    # Older API: use best_ntree_limit
    try:
        return model.predict_proba(X, ntree_limit=model.best_ntree_limit)[:, 1]
    except Exception:
        pass
    # Fallback
    return model.predict_proba(X)[:, 1]


# -------------------- Plotting --------------------
def plot_pr_curve(y_true, y_score, out):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    plt.figure()
    plt.plot(recall, precision, label="PR curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "pr_curve.png", dpi=200)
    plt.close()


def plot_roc_curve(y_true, y_score, out):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "roc_curve.png", dpi=200)
    plt.close()


def plot_calibration(y_true, y_score, n_bins, out):
    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=n_bins, strategy="quantile")
    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfectly calibrated")
    plt.xlabel("Predicted probability")
    plt.ylabel("True frequency")
    plt.title("Calibration Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "calibration_curve.png", dpi=200)
    plt.close()


def plot_confusion(cm, out, title="Confusion Matrix"):
    tn, fp, fn, tp = cm.ravel()
    plt.figure()
    plt.imshow([[tn, fp], [fn, tp]])
    plt.title(title)
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    for (i, j, val) in [(0, 0, tn), (0, 1, fp), (1, 0, fn), (1, 1, tp)]:
        plt.text(j, i, str(val), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out / "confusion_matrix.png", dpi=200)
    plt.close()


def plot_threshold_tradeoff(y_true, y_score, out):
    thresholds = np.linspace(0.0, 1.0, 200)
    precs, recs, f1s = [], [], []
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        prec = precision_score(y_true, y_pred, zero_division=0) if y_pred.sum() > 0 else 0.0
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        precs.append(prec); recs.append(rec); f1s.append(f1)
    plt.figure()
    plt.plot(thresholds, precs, label="Precision")
    plt.plot(thresholds, recs, label="Recall")
    plt.plot(thresholds, f1s, label="F1")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Precision/Recall/F1 vs Threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "threshold_tradeoff.png", dpi=200)
    plt.close()


def plot_feature_importance_gain(model, feature_names, out):
    booster = model.get_booster()
    score = booster.get_score(importance_type="gain")
    imp = [(fname, score.get(f"f{idx}", 0.0)) for idx, fname in enumerate(feature_names)]
    imp_sorted = sorted(imp, key=lambda x: x[1], reverse=True)[:25]
    labels = [k for k, _ in imp_sorted]
    vals = [v for _, v in imp_sorted]
    plt.figure(figsize=(8, max(4, len(labels) * 0.25)))
    plt.barh(range(len(labels)), vals)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Gain")
    plt.title("Top Feature Importances (gain)")
    plt.tight_layout()
    plt.savefig(out / "feature_importance_gain.png", dpi=200)
    plt.close()


def maybe_shap_summary(model, X_sample, feature_names, out):
    if not SHAP_AVAILABLE:
        return False
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        plt.figure()
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(out / "shap_summary.png", dpi=200, bbox_inches="tight")
        plt.close()
        return True
    except Exception:
        return False


# -------------------- Main --------------------
def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect dataset if not provided
    if args.data is None:
        for cand in [Path("credit_card.csv"), Path("creditcard.csv"), Path("credit-card.csv"), Path("CreditCard.csv")]:
            if cand.exists():
                args.data = str(cand)
                print(f"[info] Auto-detected dataset: {cand}")
                break
        if args.data is None:
            here = Path(__file__).parent
            for cand in [here / "credit_card.csv", here / "creditcard.csv", here / "credit-card.csv", here / "CreditCard.csv"]:
                if cand.exists():
                    args.data = str(cand)
                    print(f"[info] Auto-detected dataset: {cand}")
                    break
    if args.data is None:
        raise SystemExit("No --data provided and no local CSV found (looked for credit_card.csv / creditcard.csv).")

    df = pd.read_csv(args.data)

    # Auto-detect target
    if args.target is None:
        for cand in ["Class", "isFraud", "fraud", "Fraud", "Approved", "approved", "target", "label"]:
            if cand in df.columns:
                args.target = cand
                print(f"[info] Auto-detected target column: {args.target}")
                break
        if args.target is None:
            raise SystemExit(f"Could not infer target column from {df.columns.tolist()[:20]} ... Please pass --target <colname>.")

    # y / X
    y = to_binary(df[args.target])
    X_df = df.drop(columns=[args.target])

    # Feature types
    cat_cols = [c for c in X_df.columns if str(X_df[c].dtype) in ("object", "category", "bool")]
    num_cols = [c for c in X_df.columns if c not in cat_cols]

    # OneHotEncoder: handle both old and new sklearn
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)  # sklearn >=1.2
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)         # sklearn <1.2

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", ohe, cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    # Split data
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    X_tr_df, X_val_df, y_tr, y_val = train_test_split(
        X_train_df, y_train, test_size=0.2, random_state=args.seed, stratify=y_train
    )

    # Fit preprocessor and transform
    X_tr = preprocess.fit_transform(X_tr_df)
    X_val = preprocess.transform(X_val_df)
    X_test = preprocess.transform(X_test_df)

    # Feature names after preprocessing
    try:
        cat_names = list(preprocess.named_transformers_["cat"].get_feature_names_out(cat_cols)) if cat_cols else []
        feature_names = cat_names + num_cols
    except Exception:
        feature_names = [f"f{i}" for i in range(X_test.shape[1])]

    # Class imbalance weight
    spw = compute_scale_pos_weight(y_train)

    # Build model
    clf = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        max_depth=args.max_depth,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_weight=args.min_child_weight,
        gamma=args.gamma,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        scale_pos_weight=spw,
        random_state=args.seed,
        n_jobs=0
    )

    # Train with early stopping (version-agnostic)
    fit_kwargs = dict(eval_set=[(X_val, y_val)], verbose=False)
    trained = False
    try:
        es_cb = xgb.callback.EarlyStopping(rounds=50, save_best=True, data_name="validation_0", metric_name="auc")
        clf.fit(X_tr, y_tr, callbacks=[es_cb], **fit_kwargs)
        trained = True
    except Exception:
        pass
    if not trained:
        try:
            clf.fit(X_tr, y_tr, early_stopping_rounds=50, **fit_kwargs)
            trained = True
        except TypeError:
            pass
    if not trained:
        clf.fit(X_tr, y_tr, **fit_kwargs)

    # Predict, threshold, metrics
    y_proba = predict_proba_best(clf, X_test)
    thr, f1b, precb, recb = choose_best_threshold(y_test, y_proba, beta=1.0)
    y_pred = (y_proba >= thr).astype(int)

    auroc = roc_auc_score(y_test, y_proba)
    auprc = average_precision_score(y_test, y_proba)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    # Save metrics
    metrics = {
        "class_ratio": {
            "train_pos": int(np.sum(y_train == 1)),
            "train_neg": int(np.sum(y_train == 0)),
            "test_pos": int(np.sum(y_test == 1)),
            "test_neg": int(np.sum(y_test == 0)),
            "scale_pos_weight": spw
        },
        "threshold": thr,
        "f1_at_threshold": f1b,
        "precision_at_threshold": precb,
        "recall_at_threshold": recb,
        "auroc": auroc,
        "auprc": auprc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm.tolist()
    }
    with open(Path(out_dir) / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Plots
    plot_pr_curve(y_test, y_proba, out_dir)
    plot_roc_curve(y_test, y_proba, out_dir)
    plot_calibration(y_test, y_proba, args.calibration_bins, out_dir)
    plot_confusion(cm, out_dir)
    plot_threshold_tradeoff(y_test, y_proba, out_dir)
    plot_feature_importance_gain(clf, feature_names, out_dir)

    # SHAP (optional)
    used_shap = False
    if X_test.shape[0] > 0:
        n_sample = min(2000, X_test.shape[0])
        used_shap = maybe_shap_summary(clf, X_test[:n_sample], feature_names, out_dir)

    with open(Path(out_dir) / "run_report.json", "w") as f:
        json.dump({"used_shap": used_shap, "outputs_dir": str(Path(out_dir).resolve())}, f, indent=2)

    print("Training complete.")
    print(json.dumps(metrics, indent=2))
    print(f"Artifacts written to: {Path(out_dir).resolve()}")
    if not used_shap:
        print("Note: SHAP not generated (package missing or error); install 'shap' to enable.")


if __name__ == "__main__":
    main()
