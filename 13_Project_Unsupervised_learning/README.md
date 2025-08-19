# Unsupervised EV Clustering — Ready-to-Run Project

This project performs **clustering and dimensionality reduction** on the Kaggle dataset:
**Electric Vehicle Population Data 2024** (Utkarsh Singh).

- Kaggle dataset page: (search "Electric Vehicle Population Data 2024" by Utkarsh Singh on Kaggle)
- Save the CSV to `data/ev_population.csv` before running the script.

## What you get
- PCA → **K-Means / GMM / Agglomerative** variants (≥3).
- Internal metrics: **Silhouette**, **Davies–Bouldin**, **Calinski–Harabasz**.
- **Stability** via bootstrap **Adjusted Rand Index (ARI)**.
- Plots: elbow, silhouette, PCA scatter coloured by clusters.
- Cluster profiling tables saved to `reports/`.
- One-file runner: `python src/unsupervised_ev.py`.

## Quickstart

1) **Install dependencies** (Python 3.9+ recommended):
```bash
pip install -r requirements.txt
```

2) **Download the data** from Kaggle and place it at:
```
data/ev_population.csv
```
(You can download any of these equivalent Washington State EV population datasets on Kaggle.)

3) **Run**:
```bash
python src/unsupervised_ev.py --csv data/ev_population.csv --out reports --max-k 8
```
Optional flags:
- `--use-umap` to add a UMAP visual (requires `umap-learn`).
- `--bootstrap 20` to change stability iterations.
- `--sample 5000` to subsample rows for speed.

## Notes
- Script auto-detects columns commonly present in WA EV datasets (e.g., `Model Year`, `Make`, `Model`, `Electric Range`, `Base MSRP`, `Electric Vehicle Type`, `CAFV Eligibility`, `County`, `City`). Missing columns are handled gracefully.
- Engineered features include **vehicle age** and **range per $** when MSRP is available.
- Results are saved under `reports/` (CSV + PNG).

