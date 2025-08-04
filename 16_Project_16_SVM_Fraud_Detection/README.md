Credit-Card Fraud Detection with Linear-SVM

> **Dataset:** Kaggle “Credit Card Fraud Detection” (European card transactions, Sept 2013).  
> **Records:** 284 807 transactions   |   **Fraud rate:** ~0.17 % (492 frauds)

Project goal
Detect fraudulent card transactions **fast** while providing clear, portfolio-quality evaluation graphics (ROC, PR, confusion matrix).  The pipeline needs to run on a laptop in ≤ 30 s yet remain extensible to heavier, non-linear models.

Results (default threshold 0.50)
| Metric | Value |
|--------|-------|
| **ROC-AUC** | **0.980** |
| **PR-AUC** | **0.700** |
| **Recall** | 55 % |
| **Precision** | 87 % |

**Confusion matrix** (see `figures/confusion_matrix.png`):

The model ranks transactions very well (high ROC‑AUC) but, at the neutral 0.50 threshold, misses ≈ 45 % of frauds.  Lowering the threshold to ≈ 0.30 lifts recall > 75 % with tolerable false alerts.

Repository layout
```
├── Project-SVM.py          ← Main script (linear SVM, CV, graphics)
├── creditcard.csv          ← Dataset (add yourself or symlink)
├── figures/                ← Auto‑generated PNGs on each run
│   ├── roc_curve.png
│   ├── pr_curve.png
│   └── confusion_matrix.png
└── README.md               ← This file
```

How to run
```bash
Create venv & install deps
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install scikit-learn tqdm seaborn matplotlib pandas numpy

Execute (defaults: ./creditcard.csv → ./figures)
python Project-SVM.py

 Credit & license
*Author:* Sam 
*License:* MIT.  Dataset is provided under the original Kaggle terms.
