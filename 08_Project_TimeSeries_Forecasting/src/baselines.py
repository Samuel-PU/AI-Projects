"""
baselines.py
------------
Naive, seasonal naive, and moving-average baselines.
All functions accept a 1D numpy series and return a forecast aligned
to the last len(series) values (you'll usually slice to match y_test).
"""
import numpy as np

def naive_forecast(series: np.ndarray) -> np.ndarray:
    """ŷ_t = y_{t-1} (first element becomes nan)."""
    y = np.asarray(series).astype(float)
    yhat = np.empty_like(y); yhat[:] = np.nan
    yhat[1:] = y[:-1]
    return yhat

def seasonal_naive(series: np.ndarray, season: int) -> np.ndarray:
    """ŷ_t = y_{t-season}; first 'season' elements become nan."""
    y = np.asarray(series).astype(float)
    yhat = np.empty_like(y); yhat[:] = np.nan
    if season <= 0 or season >= len(y):
        return yhat
    yhat[season:] = y[:-season]
    return yhat

def moving_average(series: np.ndarray, window: int = 5) -> np.ndarray:
    """Centered moving average; edges -> nan."""
    y = np.asarray(series).astype(float)
    if window < 1 or window > len(y):
        return np.full_like(y, np.nan)
    cumsum = np.cumsum(np.insert(y, 0, 0.0))
    # trailing MA (past window)
    out = np.full_like(y, np.nan)
    out[window-1:] = (cumsum[window:] - cumsum[:-window]) / window
    return out
