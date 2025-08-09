"""
utils.py
--------
Shared helpers: seeding, metrics, plotting theme.
"""
import os, random
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt



def load_dataset(npz_path):
    """
    Load dataset from .npz file created by prepare_windows.py.
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    data = np.load(npz_path)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val   = data["X_val"]
    y_val   = data["y_val"]
    X_test  = data["X_test"]
    y_test  = data["y_test"]
    return X_train, y_train, X_val, y_val, X_test, y_test
def set_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def set_plot_theme():
    # Clean seaborn theme for all plots
    sns.set_theme(context="notebook", style="whitegrid")

def rmse(y, yhat) -> float:
    y = np.asarray(y); yhat = np.asarray(yhat)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))

def mae(y, yhat) -> float:
    y = np.asarray(y); yhat = np.asarray(yhat)
    return float(np.mean(np.abs(y - yhat)))

def mape(y, yhat) -> float:
    y = np.asarray(y); yhat = np.asarray(yhat)
    return float(np.mean(np.abs((y - yhat) / (y + 1e-8))) * 100.0)
