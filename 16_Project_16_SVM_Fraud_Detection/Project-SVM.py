#!/usr/bin/env python3
"""
Credit-Card Fraud Detection – Linear-SVM with Progress Bar & Seaborn Graphics
============================================================================
*Fixed*: `_save()` no longer crashes when the output directory is **relative**.
Just run:

```bash
python Project-SVM.py          # uses ./creditcard.csv → ./figures/*.png
```

Pass `--csv` or `--out` to override. Figures are saved; none pop up on screen.
"""

from __future__ import annotations

import argparse
import time
import pathlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")  # must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC  # swap to SVC(kernel="rbf") for more accuracy
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    confusion_matrix,
)
from tqdm.auto import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, path: pathlib.Path) -> None:
    """Save `fig` to *path* and report relative path if possible."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    try:
        rel = path.resolve().relative_to(pathlib.Path.cwd())
    except ValueError:  # path is on another drive or outside CWD
        rel = path.resolve()
    tqdm.write(f"saved → {rel}")
    plt.close(fig)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Linear-SVM fraud detector")
    p.add_argument("--csv", type=pathlib.Path, default="creditcard.csv",
                   help="CSV file (default: ./creditcard.csv)")
    p.add_argument("--out", type=pathlib.Path, default="figures",
                   help="directory for PNGs (default: ./figures)")
    p.add_argument("--seed", type=int, default=42, help="random seed")
    p.add_argument("--folds", type=int, default=5, help="CV folds")
    return p

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parser().parse_args()
    t0 = time.perf_counter()

    # 1. Load data ---------------------------------------------------------------
    if not args.csv.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv.resolve()}")
    df = pd.read_csv(args.csv)
    if "Class" not in df.columns:
        raise ValueError("CSV must contain a 'Class' column with labels 0/1")
    y = df["Class"].astype(int).values
    X = df.drop(columns=["Class"])

    # 2. Model -------------------------------------------------------------------
    base = LinearSVC(C=1.0, class_weight="balanced", random_state=args.seed)
    svm = make_pipeline(
        StandardScaler(),
        CalibratedClassifierCV(base, cv=3, method="sigmoid"),
    )

    # 3. Stratified CV with progress bar -----------------------------------------
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof = np.zeros_like(y, dtype=float)
    for fold, (train_idx, test_idx) in enumerate(
        tqdm(cv.split(X, y), total=cv.get_n_splits(), desc="CV folds"), start=1
    ):
        svm.fit(X.iloc[train_idx], y[train_idx])
        proba = svm.predict_proba(X.iloc[test_idx])[:, 1]
        oof[test_idx] = proba
        auc = roc_auc_score(y[test_idx], proba)
        prc = average_precision_score(y[test_idx], proba)
        tqdm.write(f"  fold {fold}: ROC-AUC={auc:.4f}  PR-AUC={prc:.4f}")

    # 4. Aggregate metrics --------------------------------------------------------
    roc_all = roc_auc_score(y, oof)
    pr_all = average_precision_score(y, oof)
    print(f"\nOverall ROC-AUC={roc_all:.4f}   PR-AUC={pr_all:.4f}")

    sns.set_theme(style="whitegrid", font_scale=1.2)

    # 5-a ROC curve
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    RocCurveDisplay.from_predictions(y, oof, ax=ax1)
    ax1.set_title("SVM – ROC curve")
    _save(fig1, pathlib.Path(args.out) / "roc_curve.png")

    # 5-b PR curve
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    PrecisionRecallDisplay.from_predictions(y, oof, ax=ax2)
    ax2.set_title("SVM – Precision-Recall curve")
    _save(fig2, pathlib.Path(args.out) / "pr_curve.png")

    # 5-c Confusion matrix
    y_hat = (oof >= 0.5).astype(int)
    cm = confusion_matrix(y, y_hat)
    fig3, ax3 = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax3,
                xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
    ax3.set_title("Confusion matrix (thr = 0.50)")
    ax3.set_xlabel("Predicted"); ax3.set_ylabel("Actual")
    _save(fig3, pathlib.Path(args.out) / "confusion_matrix.png")

    # 6. Summary -----------------------------------------------------------------
    mins = (time.perf_counter() - t0) / 60
    print(f"\n✓ Finished in {mins:.1f} min – figures in {pathlib.Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
