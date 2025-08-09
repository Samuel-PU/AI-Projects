# ============================================
# Stroke Dataset: OLS vs Spline+ElasticNet vs XGBoost (Regression)
# Target: avg_glucose_level
# Outputs: PNG charts + CSV metrics + text summary in ./output
# ============================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, SplineTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, make_scorer

# XGBoost
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

# -----------------------------
# Config
# -----------------------------
DATA_PATH = "stroke_data.csv"        # update if needed

TARGET = "avg_glucose_level"
ID_COLS = ["id"]
DROP_COLS = ["stroke"]               # regression task

RANDOM_STATE = 42
TEST_SIZE = 0.2

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# -----------------------------
# Helper: safe savefig
# -----------------------------
def savefig(path, tight=True, dpi=200):
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()

# -----------------------------
# Load & clean
# -----------------------------
df = pd.read_csv(DATA_PATH)
df.columns = [c.strip() for c in df.columns]

for col in ["bmi", "avg_glucose_level", "age"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

for c in ID_COLS + DROP_COLS:
    if c in df.columns:
        df = df.drop(columns=c)

keep_cols = [
    "gender", "age", "hypertension", "heart_disease",
    "ever_married", "work_type", "Residence_type",
    "avg_glucose_level", "bmi", "smoking_status"
]
df = df[[c for c in keep_cols if c in df.columns]].copy()

y = df[TARGET].copy()
feature_cols = [c for c in df.columns if c != TARGET]
X = df[feature_cols].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

numeric_feats = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
categorical_feats = [c for c in X.columns if c not in numeric_feats]

# -----------------------------
# Preprocessors
# -----------------------------
# Baseline preprocessor (linear models)
baseline_pre = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric_feats),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), categorical_feats)
    ],
    remainder="drop",
    verbose_feature_names_out=False
)

# Advanced preprocessor with splines for age/bmi
spline_numeric_feats = [c for c in ["age", "bmi"] if c in numeric_feats]
plain_numeric_feats = [c for c in numeric_feats if c not in spline_numeric_feats]

advanced_pre = ColumnTransformer(
    transformers=[
        ("num_spline", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("spline", SplineTransformer(degree=3, n_knots=7, include_bias=False)),
            ("scaler", StandardScaler())
        ]), spline_numeric_feats),
        ("num_plain", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), plain_numeric_feats),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), categorical_feats),
    ],
    remainder="drop",
    verbose_feature_names_out=False
)

# Tree-friendly preprocessor (no scaling; still one-hot for safety)
tree_pre = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), numeric_feats),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), categorical_feats),
    ],
    remainder="drop",
    verbose_feature_names_out=False
)

# -----------------------------
# Models
# -----------------------------
baseline_model = Pipeline(steps=[
    ("prep", baseline_pre),
    ("ols", LinearRegression())
])

advanced_model = Pipeline(steps=[
    ("prep", advanced_pre),
    ("enet", ElasticNetCV(
        alphas=np.logspace(-3, 2, 60),
        l1_ratio=[.15, .3, .5, .7, .85, .95],
        cv=5,
        max_iter=20000,
        n_jobs=None,
        random_state=RANDOM_STATE
    ))
])

xgb_model = None
if XGB_AVAILABLE:
    xgb_model = Pipeline(steps=[
        ("prep", tree_pre),
        ("xgb", XGBRegressor(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            objective="reg:squarederror",
            eval_metric="rmse",
            tree_method="hist"   # fast/default; set "gpu_hist" if you have CUDA
        ))
    ])

# -----------------------------
# Cross-validated comparison (train)
# -----------------------------
rkf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=RANDOM_STATE)
rmse_scorer = make_scorer(lambda yt, yp: mean_squared_error(yt, yp, squared=False), greater_is_better=False)

cv_rows = []
for name, model in [
    ("OLS (Baseline)", baseline_model),
    ("Spline + ElasticNet", advanced_model),
    ("XGBoost", xgb_model if xgb_model else None)
]:
    if model is None:
        continue
    scores = cross_val_score(model, X_train, y_train, cv=rkf, scoring="neg_root_mean_squared_error")
    cv_rows.append({
        "model": name,
        "cv_rmse_mean": -scores.mean(),
        "cv_rmse_std": scores.std()
    })

cv_table = pd.DataFrame(cv_rows).sort_values("cv_rmse_mean")
cv_table.to_csv(OUTPUT_DIR / "model_cv_comparison.csv", index=False)

# -----------------------------
# Fit on full train & evaluate on test
# -----------------------------
def evaluate(model, Xtr, ytr, Xte, yte):
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    return {
        "rmse": mean_squared_error(yte, pred, squared=False),
        "mae": mean_absolute_error(yte, pred),
        "r2": r2_score(yte, pred),
        "y_pred": pred
    }

results = []
name_to_pred = {}

for name, model in [
    ("OLS (Baseline)", baseline_model),
    ("Spline + ElasticNet", advanced_model),
]:
    res = evaluate(model, X_train, y_train, X_test, y_test)
    results.append({"model": name, "rmse": res["rmse"], "mae": res["mae"], "r2": res["r2"]})
    name_to_pred[name] = res["y_pred"]

if xgb_model is not None:
    res_xgb = evaluate(xgb_model, X_train, y_train, X_test, y_test)
    results.append({"model": "XGBoost", "rmse": res_xgb["rmse"], "mae": res_xgb["mae"], "r2": res_xgb["r2"]})
    name_to_pred["XGBoost"] = res_xgb["y_pred"]

metrics_table = pd.DataFrame(results).sort_values("rmse")
metrics_table.to_csv(OUTPUT_DIR / "model_test_metrics.csv", index=False)

print("\nCross-validated RMSE (train):")
print(cv_table.to_string(index=False))
print("\nTest metrics:")
print(metrics_table.to_string(index=False))

# -----------------------------
# Visuals (Seaborn) – saved to ./output
# -----------------------------
sns.set_theme(style="whitegrid")

# 1) RMSE comparison (test)
plt.figure(figsize=(7.2, 4.2))
ax = sns.barplot(data=metrics_table, x="model", y="rmse")
ax.set_title("Test RMSE (Lower is Better)")
ax.set_xlabel("")
ax.set_ylabel("RMSE")
for p in ax.patches:
    ax.annotate(f"{p.get_height():.2f}",
                (p.get_x() + p.get_width()/2., p.get_height()),
                ha='center', va='bottom', fontsize=10, xytext=(0, 3), textcoords='offset points')
savefig(OUTPUT_DIR / "01_rmse_comparison.png")

# 2/3/… Predicted vs Actual + Residuals for each model
def pred_actual_and_residuals(name, y_true, y_pred, prefix):
    # Pred vs Actual
    plt.figure(figsize=(5.6, 5.6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, linestyle="--")
    row = metrics_table[metrics_table["model"] == name].iloc[0]
    plt.title(f"{name}: Predicted vs Actual\nRMSE={row['rmse']:.2f}, R²={row['r2']:.3f}")
    plt.xlabel("Actual avg_glucose_level")
    plt.ylabel("Predicted")
    savefig(OUTPUT_DIR / f"{prefix}_pred_vs_actual.png")

    # Residuals vs Fitted
    resid = y_true - y_pred
    plt.figure(figsize=(6.2, 4.2))
    sns.scatterplot(x=y_pred, y=resid, alpha=0.6)
    plt.axhline(0, linestyle="--")
    plt.title(f"{name}: Residuals vs Fitted")
    plt.xlabel("Fitted (Predicted)")
    plt.ylabel("Residual")
    savefig(OUTPUT_DIR / f"{prefix}_residuals_vs_fitted.png")

pred_actual_and_residuals("OLS (Baseline)", y_test, name_to_pred["OLS (Baseline)"], "02_OLS")
pred_actual_and_residuals("Spline + ElasticNet", y_test, name_to_pred["Spline + ElasticNet"], "03_SplineENet")
if "XGBoost" in name_to_pred:
    pred_actual_and_residuals("XGBoost", y_test, name_to_pred["XGBoost"], "04_XGBoost")

# 6) Effect curve helper (1D): vary var, hold others at typical values (advanced model)
def typical_row(X_df):
    row = {}
    for c in X_df.columns:
        if pd.api.types.is_numeric_dtype(X_df[c]):
            row[c] = X_df[c].median()
        else:
            m = X_df[c].mode(dropna=True)
            row[c] = m.iloc[0] if not m.empty else X_df[c].iloc[0]
    return pd.DataFrame([row])

def effect_curve_1d(model, X_df, var, n=100):
    base = typical_row(X_df)
    vmin, vmax = np.nanpercentile(X_df[var], [5, 95])
    grid = np.linspace(vmin, vmax, n)
    rows = []
    for g in grid:
        r = base.copy()
        r[var] = g
        rows.append(r)
    Xg = pd.concat(rows, ignore_index=True)
    preds = model.predict(Xg)
    return grid, preds

if "age" in X.columns:
    age_grid, age_preds = effect_curve_1d(advanced_model, X_train, "age", n=120)
    plt.figure(figsize=(6.2, 4.2))
    sns.lineplot(x=age_grid, y=age_preds)
    plt.title("Estimated Effect on avg_glucose_level: Age\n(Spline + ElasticNet, others at typical values)")
    plt.xlabel("Age")
    plt.ylabel("Predicted avg_glucose_level")
    savefig(OUTPUT_DIR / "06_effect_curve_age.png")

if "bmi" in X.columns and not X_train["bmi"].dropna().empty:
    bmi_grid, bmi_preds = effect_curve_1d(advanced_model, X_train, "bmi", n=120)
    plt.figure(figsize=(6.2, 4.2))
    sns.lineplot(x=bmi_grid, y=bmi_preds)
    plt.title("Estimated Effect on avg_glucose_level: BMI\n(Spline + ElasticNet, others at typical values)")
    plt.xlabel("BMI")
    plt.ylabel("Predicted avg_glucose_level")
    savefig(OUTPUT_DIR / "07_effect_curve_bmi.png")

# 7) XGBoost feature importance (if available)
def get_feature_names_from_preprocessor(prep, input_cols):
    """Map ColumnTransformer to flat feature names (works for pipelines used above)."""
    names = []
    for name, trans, cols in prep.transformers_:
        if name == 'remainder' and trans == 'drop':
            continue
        if hasattr(trans, 'named_steps'):
            last = list(trans.named_steps.values())[-1]
        else:
            last = trans
        if hasattr(last, 'get_feature_names_out'):
            # supply original column names where possible
            try:
                feats = last.get_feature_names_out(cols)
            except Exception:
                feats = last.get_feature_names_out()
            names.extend(feats)
        else:
            # passthrough columns
            if isinstance(cols, (list, tuple, np.ndarray)):
                names.extend(cols)
            elif cols == 'remainder':
                remaining = [c for c in input_cols if c not in sum(
                    [[c for c in ct[2]] for ct in prep.transformers_ if ct[2] != 'remainder'], []
                )]
                names.extend(remaining)
    return np.array(names, dtype=str)

if xgb_model is not None:
    # fit once on all training data to extract importances
    xgb_model.fit(X_train, y_train)
    feature_names = get_feature_names_from_preprocessor(xgb_model.named_steps["prep"], X_train.columns)
    booster = xgb_model.named_steps["xgb"].get_booster()
    scores = booster.get_score(importance_type="gain")  # dict: feature -> importance
    # XGBoost indexes features as f0, f1, ...
    imp = []
    for i, fname in enumerate(feature_names):
        key = f"f{i}"
        imp.append((fname, scores.get(key, 0.0)))
    imp_df = pd.DataFrame(imp, columns=["feature", "gain"]).sort_values("gain", ascending=False).head(25)
    imp_df.to_csv(OUTPUT_DIR / "xgb_top25_feature_importance.csv", index=False)

    plt.figure(figsize=(8.5, 7))
    sns.barplot(data=imp_df, x="gain", y="feature")
    plt.title("XGBoost Feature Importance (Top 25 by Gain)")
    plt.xlabel("Gain")
    plt.ylabel("")
    savefig(OUTPUT_DIR / "08_xgb_feature_importance.png")

# -----------------------------
# Human-friendly summary file
# -----------------------------
best_row = metrics_table.iloc[0]
summary_lines = [
    "Stroke Dataset — Regression Results",
    "===================================",
    "",
    f"Target: {TARGET}",
    "",
    "Models compared:",
    "• OLS (Baseline)",
    "• Spline + ElasticNet (adds smooth non-linearities + regularisation)",
    ("• XGBoost (gradient-boosted trees)" if xgb_model is not None else "• XGBoost not available (package missing)"),
    "",
    "Cross-validated RMSE on training (lower is better):",
    cv_table.to_string(index=False),
    "",
    "Test set performance:",
    metrics_table.to_string(index=False),
    "",
    f"Best on test: {best_row['model']} (RMSE={best_row['rmse']:.2f}, R²={best_row['r2']:.3f})",
    "",
    "How to read the charts:",
    "1) 01_rmse_comparison.png — shorter bar is better.",
    "2) 02/03/04_*_pred_vs_actual.png — closer to the dashed diagonal = better predictions.",
    "3) 02/03/04_*_residuals_vs_fitted.png — residuals should hover around zero without a funnel shape.",
    "4) 06/07_effect_curve_*.png — how age/BMI (holding others typical) affect predicted glucose (Spline+ENet).",
    "5) 08_xgb_feature_importance.png — top features by gain (if XGBoost available).",
    "",
    "Notes:",
    "• Splines capture curved relationships in age/BMI while ElasticNet controls complexity.",
    "• XGBoost can capture higher-order interactions; compare metrics and importances.",
]

with open(OUTPUT_DIR / "README_results.txt", "w") as f:
    f.write("\n".join(summary_lines))

print(f"\nAll done. See the '{OUTPUT_DIR}/' folder for charts and metrics.")
