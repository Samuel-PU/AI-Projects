# Stroke Dataset: Predicting Average Glucose Levels

This project explores and compares three regression models to predict `avg_glucose_level` using the Stroke Prediction Dataset. The focus is on model interpretability, diagnostic accuracy, and performance benchmarking.

## Dataset

- Source: Stroke Prediction Dataset
- Target variable: `avg_glucose_level`
- Features used: demographic and health-related attributes (e.g. age, bmi, smoking status)

## Models Compared

1. **OLS (Baseline)** — Standard linear regression model
2. **Spline + ElasticNet** — Regularised model with spline-transformed non-linear features
3. **XGBoost** — Tree-based gradient boosting model with automated feature interactions

## Preprocessing

- **Missing values**:
  - Numeric: imputed with median
  - Categorical: imputed with mode
- **Encoding**:
  - Categorical features: one-hot encoded
- **Feature scaling**:
  - Applied to linear models only (OLS, ElasticNet)

## Evaluation Metrics

Performance was assessed on a held-out test set using:

- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² (coefficient of determination)

## Results

### Cross-Validated RMSE (Training)

| Model               | CV RMSE (Mean) | CV RMSE (Std) |
|--------------------|----------------|---------------|
| OLS (Baseline)      | ~42.04         | ±1.27         |
| Spline + ElasticNet | ~42.13         | ±1.30         |
| XGBoost             | ~43.50         | ±1.40         |

### Test Set Performance

| Model               | RMSE   | MAE   | R²     |
|--------------------|--------|-------|--------|
| OLS (Baseline)      | 42.29  | ~30   | 0.070  |
| Spline + ElasticNet | 42.45  | ~30   | 0.063  |
| XGBoost             | 44.09  | ~31   | -0.011 |

### Key Findings

- The **OLS baseline** slightly outperformed both advanced models in terms of test RMSE and R².
- The **Spline + ElasticNet** model captured meaningful non-linear effects (particularly for age and BMI), but this did not translate into better generalisation performance.
- **XGBoost underperformed** the linear models despite offering a strong feature ranking. Its predictions were more dispersed and resulted in a negative R² on the test set.

## Visual Diagnostics

All plots are saved to the `output/` folder:

1. `01_rmse_comparison.png` — RMSE comparison bar chart
2. `02_OLS_pred_vs_actual.png` — OLS predicted vs actual
3. `02_OLS_residuals_vs_fitted.png` — OLS residual diagnostics
4. `03_SplineENet_pred_vs_actual.png` — ElasticNet predicted vs actual
5. `03_SplineENet_residuals_vs_fitted.png` — ElasticNet residuals
6. `04_XGBoost_pred_vs_actual.png` — XGBoost predictions
7. `04_XGBoost_residuals_vs_fitted.png` — XGBoost residuals
8. `06_effect_curve_age.png` — Marginal effect of age
9. `07_effect_curve_bmi.png` — Marginal effect of BMI
10. `08_xgb_feature_importance.png` — Top 25 XGBoost features by gain

## Interpretation of Plots

- **Predicted vs Actual**: All models underpredict high glucose levels (above ~150), suggesting missing explanatory features or inherent noise in the data.
- **Residuals vs Fitted**: All models exhibit heteroscedasticity, with structured residual patterns indicating non-random error — a sign of model misfit.
- **Effect Curves (Spline + ENet)**:
  - Age: U-shaped relationship with glucose
  - BMI: Monotonically increasing effect
- **Feature Importance (XGBoost)**:
  - Top features: `hypertension`, `heart_disease`, `age`, `bmi`, `ever_married`, `smoking_status`

## Limitations

- Low R² values across all models suggest that the available features only weakly explain variation in glucose levels.
- XGBoost may be overfitting or suffering from data imbalance.
- Feature interactions and latent medical conditions not captured in this dataset may significantly affect glucose levels.

## Recommendations

To improve model performance:

- Engineer interaction features (e.g. `age × bmi`)
- Consider log-transforming the target (`avg_glucose_level`)
- Try alternative models like LightGBM or Huber regression
- Explore clustering or anomaly detection for outlier management

## Requirements

- Python 3.8+
- pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost

Install dependencies with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
