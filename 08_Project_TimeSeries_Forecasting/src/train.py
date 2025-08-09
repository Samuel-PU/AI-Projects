# train.py
import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, GRU, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns

from .utils import load_dataset

# Custom RMSE metric
def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

def build_lstm(win):
    inputs = Input(shape=(win, 1))
    x = LSTM(64, activation="tanh")(inputs)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    return model

def build_gru(win):
    inputs = Input(shape=(win, 1))
    x = GRU(64, activation="tanh")(inputs)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    return model

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["lstm", "gru"], required=True)
    p.add_argument("--win", type=int, required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--data", default="data/processed/dataset.npz")
    p.add_argument("--out", default="output")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    X_train, y_train, X_val, y_val, _, _ = load_dataset(args.data)

    if args.model == "lstm":
        model = build_lstm(args.win)
    else:
        model = build_gru(args.win)

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae", "mse", rmse]
    )

    checkpoint_path = os.path.join(args.out, f"{args.model}_best.keras")
    callbacks = [
        ModelCheckpoint(checkpoint_path, save_best_only=True, monitor="val_loss"),
        EarlyStopping(patience=5, restore_best_weights=True)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch,
        callbacks=callbacks,
        verbose=1
    )

    # Plot training history
    plots_dir = os.path.join(args.out, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    for metric in ["loss", "mae", "mse", "rmse"]:
        if metric in history.history:
            sns.lineplot(x=range(len(history.history[metric])), y=history.history[metric], label=f"Train {metric}")
            sns.lineplot(x=range(len(history.history[f"val_{metric}"])), y=history.history[f"val_{metric}"], label=f"Val {metric}")
    plt.xlabel("Epochs")
    plt.ylabel("Metric Value")
    plt.title(f"{args.model.upper()} Training History")
    plt.legend()
    plt.savefig(os.path.join(plots_dir, f"{args.model}_training_history.png"))
    plt.close()

if __name__ == "__main__":
    main()
