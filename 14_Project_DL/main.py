#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text Classification — End-to-End Project (Binary)
-------------------------------------------------
Models:
  1) Baseline: TF-IDF + Logistic Regression (scikit-learn)
  2) TextCNN (Keras)
  3) BiLSTM (Keras)
  4) DistilBERT fine-tuning (HuggingFace Transformers)

Usage (examples):
  # Use IMDB (auto-downloads; no accounts needed)
  python text_classification_project.py --dataset imdb --models baseline,lstm,textcnn,distilbert --out ./reports

  # Use your own CSV with columns: text,target (0/1)
  python text_classification_project.py --dataset csv --train_csv /path/to/data.csv --models baseline,lstm --out ./reports

Requirements:
  pip install datasets scikit-learn matplotlib tensorflow transformers torch
"""

import argparse, os, re, html, json, random, time, warnings
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"


# Silence TF/transformers warnings until used
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=FutureWarning)

# =========
# Utilities
# =========

def now():
    return time.strftime("%H:%M:%S")

def log(msg: str):
    print(f"[{now()}] {msg}", flush=True)

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

URL_RE = re.compile(r"http[s]?://\S+")
MENTION_RE = re.compile(r"@[A-Za-z0-9_]+")
HASHTAG_RE = re.compile(r"#([A-Za-z0-9_]+)")

def clean_text(text: str) -> str:
    """
    Minimal text cleaning for tweets/reviews:
      - unescape HTML
      - remove URLs and @mentions
      - keep hashtags as tokens (strip the #)
      - normalise whitespace
      - lowercase
    """
    if not isinstance(text, str):
        return ""
    x = html.unescape(text)
    x = URL_RE.sub(" ", x)
    x = MENTION_RE.sub(" ", x)
    x = HASHTAG_RE.sub(lambda m: " " + m.group(1) + " ", x)
    x = re.sub(r"\s+", " ", x)
    return x.strip().lower()

def plot_confusion_matrix(y_true, y_pred, out_png: str, title: str = "Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def plot_pr_roc(y_true, y_score, out_dir: str, prefix: str):
    ensure_dir(out_dir)
    # PR Curve
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    plt.figure(figsize=(5,4))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall (AP={ap:.3f})")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{prefix}_pr_curve.png"), dpi=200); plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr)
    plt.plot([0,1], [0,1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC (AUC={auc:.3f})")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{prefix}_roc_curve.png"), dpi=200); plt.close()

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

# ===============
# Data management
# ===============

def load_dataset_csv(train_csv: str, limit_n: int = 0) -> Tuple[pd.Series, pd.Series]:
    df = pd.read_csv(train_csv)
    assert "text" in df.columns and "target" in df.columns, "Expected 'text' and 'target' columns."
    df["text_clean"] = df["text"].astype(str).apply(clean_text)
    if limit_n and limit_n > 0 and len(df) > limit_n:
        df = df.sample(n=limit_n, random_state=42)
    X = df["text_clean"]
    y = df["target"].astype(int)
    return X, y

def load_dataset_imdb(limit_n: int = 0) -> Tuple[pd.Series, pd.Series]:
    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError("Please install the 'datasets' package: pip install datasets") from e

    log("Loading IMDB dataset via 'datasets'...")
    dset = load_dataset("imdb")
    texts = np.array(dset["train"]["text"])
    labels = np.array(dset["train"]["label"], dtype=int)
    if limit_n and limit_n > 0 and len(texts) > limit_n:
        idx = np.random.RandomState(42).choice(len(texts), size=limit_n, replace=False)
        texts, labels = texts[idx], labels[idx]
    texts = pd.Series([clean_text(t) for t in texts])
    labels = pd.Series(labels)
    return texts, labels

def load_data(args) -> Tuple[pd.Series, pd.Series]:
    if args.dataset.lower() == "csv":
        if not args.train_csv:
            raise ValueError("--dataset csv requires --train_csv path to a file with columns text,target")
        return load_dataset_csv(args.train_csv, limit_n=args.limit_n)
    elif args.dataset.lower() == "imdb":
        return load_dataset_imdb(limit_n=args.limit_n)
    else:
        raise ValueError("Unsupported --dataset. Choose from: imdb, csv")

# ============================
# 1) Baseline: TF-IDF + LogReg
# ============================

def run_baseline_tfidf_lr(X_train, y_train, X_val, y_val, out_dir: str) -> Dict[str, Any]:
    log("Training Baseline TF-IDF + LogisticRegression...")
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.98)
    Xtr = vec.fit_transform(X_train)
    Xva = vec.transform(X_val)

    clf = LogisticRegression(max_iter=2000)
    clf.fit(Xtr, y_train)

    prob = clf.predict_proba(Xva)[:, 1]
    pred = (prob >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_val, pred)),
        "precision": float(precision_score(y_val, pred, zero_division=0)),
        "recall": float(recall_score(y_val, pred, zero_division=0)),
        "f1": float(f1_score(y_val, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_val, prob)),
        "avg_precision": float(average_precision_score(y_val, prob)),
        "report": classification_report(y_val, pred, digits=3, zero_division=0),
    }
    ensure_dir(out_dir)
    with open(os.path.join(out_dir, "baseline_report.txt"), "w", encoding="utf-8") as f:
        f.write(metrics["report"])
    with open(os.path.join(out_dir, "baseline_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({k:v for k,v in metrics.items() if k != "report"}, f, indent=2)

    plot_confusion_matrix(y_val, pred, os.path.join(out_dir, "baseline_confusion.png"), title="Confusion (TF-IDF LR)")
    plot_pr_roc(y_val, prob, out_dir, prefix="baseline")
    return metrics

# ============================
# 2) Text models in Keras
# ============================

def build_vectoriser_keras(texts: pd.Series, vocab_size=30000, seq_len=128):
    import tensorflow as tf
    from tensorflow.keras.layers import TextVectorization

    ds = tf.data.Dataset.from_tensor_slices(texts.values).batch(256)
    vec = TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=seq_len,
        standardize=None,  # already cleaned
    )
    vec.adapt(ds)
    return vec

def load_glove_embeddings(glove_path: str, vocab: Dict[str, int], emb_dim: int) -> np.ndarray:
    embeddings_index = {}
    with open(glove_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            values = line.strip().split()
            if len(values) < emb_dim + 1:
                continue
            word = " ".join(values[:-emb_dim])
            coefs = np.asarray(values[-emb_dim:], dtype="float32")
            embeddings_index[word] = coefs

    vocab_size = len(vocab)
    emb_matrix = np.random.normal(0, 0.05, (vocab_size, emb_dim)).astype("float32")
    hits = 0
    for token, idx in vocab.items():
        vec = embeddings_index.get(token)
        if vec is not None and idx < vocab_size:
            emb_matrix[idx] = vec
            hits += 1
    log(f"GloVe: initialised {hits:,}/{vocab_size:,} tokens.")
    return emb_matrix

def run_keras_model(model_name: str, X_train, y_train, X_val, y_val, out_dir: str,
                    vocab_size=30000, seq_len=128, emb_dim=100, glove_path: Optional[str]=None,
                    batch_size=64, epochs=6, lr=1e-3) -> Dict[str, Any]:
    import tensorflow as tf
    from tensorflow.keras import layers, callbacks, optimizers, Model

    ensure_dir(out_dir)
    # Vectoriser
    vec = build_vectoriser_keras(pd.concat([X_train, X_val], axis=0), vocab_size=vocab_size, seq_len=seq_len)
    vocab = vec.get_vocabulary()
    token_to_index = {t:i for i,t in enumerate(vocab)}

    # Datasets
    AUTOTUNE = tf.data.AUTOTUNE
    def to_ds(texts, labels, shuffle=False):
        ds = tf.data.Dataset.from_tensor_slices((texts.values, labels.values))
        ds = ds.map(lambda x, y: (vec(x), tf.cast(y, tf.int32)), num_parallel_calls=AUTOTUNE)
        if shuffle:
            ds = ds.shuffle(buffer_size=2048, seed=42)
        ds = ds.batch(batch_size).prefetch(AUTOTUNE)
        return ds

    ds_tr = to_ds(X_train, y_train, shuffle=True)
    ds_va = to_ds(X_val, y_val, shuffle=False)

    # Embedding initialiser
    if glove_path:
        emb_matrix = load_glove_embeddings(glove_path, token_to_index, emb_dim)
        emb_layer = layers.Embedding(len(vocab), emb_dim, embeddings_initializer=tf.keras.initializers.Constant(emb_matrix), trainable=True)
    else:
        emb_layer = layers.Embedding(len(vocab), emb_dim)

    # Architectures
    inputs = layers.Input(shape=(seq_len,), dtype="int32")

    if model_name.lower() == "textcnn":
        x = emb_layer(inputs)
        convs = []
        for k in (3,4,5):
            c = layers.Conv1D(filters=128, kernel_size=k, padding="valid", activation="relu")(x)
            p = layers.GlobalMaxPooling1D()(c)
            convs.append(p)
        x = layers.Concatenate()(convs)
        x = layers.Dropout(0.4)(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)
    elif model_name.lower() == "lstm":
        x = emb_layer(inputs)
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
        x = layers.GlobalMaxPooling1D()(x)
        x = layers.Dropout(0.4)(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)
    else:
        raise ValueError("model_name must be 'textcnn' or 'lstm'")

    model = Model(inputs, outputs, name=model_name)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(curve="ROC", name="auc"), tf.keras.metrics.AUC(curve="PR", name="aupr")],
    )

    es = callbacks.EarlyStopping(monitor="val_aupr", mode="max", patience=2, restore_best_weights=True)
    ckpt = callbacks.ModelCheckpoint(os.path.join(out_dir, f"{model_name}_best.keras"), monitor="val_aupr", mode="max", save_best_only=True)

    history = model.fit(ds_tr, validation_data=ds_va, epochs=epochs, callbacks=[es, ckpt], verbose=2)

    # Evaluate & plot
    y_prob = model.predict(ds_va, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "precision": float(precision_score(y_val, y_pred, zero_division=0)),
        "recall": float(recall_score(y_val, y_pred, zero_division=0)),
        "f1": float(f1_score(y_val, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_val, y_prob)),
        "avg_precision": float(average_precision_score(y_val, y_prob)),
        "report": classification_report(y_val, y_pred, digits=3, zero_division=0),
        "params": model.count_params(),
    }
    with open(os.path.join(out_dir, f"{model_name}_report.txt"), "w", encoding="utf-8") as f:
        f.write(metrics["report"])
    with open(os.path.join(out_dir, f"{model_name}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({k:v for k,v in metrics.items() if k != "report"}, f, indent=2)

    plot_confusion_matrix(y_val, y_pred, os.path.join(out_dir, f"{model_name}_confusion.png"), title=f"Confusion ({model_name})")
    plot_pr_roc(y_val, y_prob, out_dir, prefix=model_name)

    # Save training history
    hist_path = os.path.join(out_dir, f"{model_name}_history.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump({k:[float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)

    return metrics

# =====================================
# 3) DistilBERT (HuggingFace Transformers)
# =====================================

@dataclass
class HFExample:
    text: str
    label: int

class HFDataset:
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = np.int64(self.labels[idx])
        return item

def run_distilbert(X_train, y_train, X_val, y_val, out_dir: str, epochs=3, batch_size=16, lr=2e-5, max_len=128) -> Dict[str, Any]:
    from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
    import torch

    ensure_dir(out_dir)

    log("Tokenising with DistilBERT tokenizer...")
    tok = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    tr_enc = tok(list(X_train.values), truncation=True, padding=True, max_length=max_len)
    va_enc = tok(list(X_val.values), truncation=True, padding=True, max_length=max_len)

    train_ds = HFDataset({k: torch.tensor(v) for k, v in tr_enc.items()}, list(y_train.values))
    val_ds   = HFDataset({k: torch.tensor(v) for k, v in va_enc.items()}, list(y_val.values))

    log("Loading DistilBERT model...")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        # Stable softmax to get class-1 probability
        exps = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = (exps / exps.sum(axis=1, keepdims=True))[:, 1]
        preds = (probs >= 0.5).astype(int)
        return {
            "accuracy": accuracy_score(labels, preds),
            "precision": precision_score(labels, preds, zero_division=0),
            "recall": recall_score(labels, preds, zero_division=0),
            "f1": f1_score(labels, preds, zero_division=0),
        }

    args = TrainingArguments(
        output_dir=os.path.join(out_dir, "hf_runs"),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=50,
        report_to=[],  # disable wandb, etc.
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    log("Fine-tuning DistilBERT...")
    trainer.train()

    # Evaluate and plot curves
    log("Evaluating DistilBERT...")
    eval_out = trainer.predict(val_ds)
    logits = eval_out.predictions
    exps = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = (exps / exps.sum(axis=1, keepdims=True))[:, 1]
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_val, preds)),
        "precision": float(precision_score(y_val, preds, zero_division=0)),
        "recall": float(recall_score(y_val, preds, zero_division=0)),
        "f1": float(f1_score(y_val, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_val, probs)),
        "avg_precision": float(average_precision_score(y_val, probs)),
        "report": classification_report(y_val, preds, digits=3, zero_division=0),
    }
    with open(os.path.join(out_dir, "distilbert_report.txt"), "w", encoding="utf-8") as f:
        f.write(metrics["report"])
    with open(os.path.join(out_dir, "distilbert_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({k:v for k,v in metrics.items() if k != "report"}, f, indent=2)

    plot_confusion_matrix(y_val, preds, os.path.join(out_dir, "distilbert_confusion.png"), title="Confusion (DistilBERT)")
    plot_pr_roc(y_val, probs, out_dir, prefix="distilbert")

    # Save final model
    trainer.save_model(os.path.join(out_dir, "distilbert_model"))
    return metrics

# ==============
# Main procedure
# ==============

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="imdb", choices=["imdb", "csv"],
                        help="Choose 'imdb' for auto-download dataset, or 'csv' to use your own file (requires --train_csv).")
    parser.add_argument("--train_csv", default="", help="Path to CSV with columns: text,target (only if --dataset csv).")
    parser.add_argument("--out", default="./reports", help="Output directory for metrics & figures.")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split (stratified).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit_n", type=int, default=0, help="Optional row limit for quick runs. 0 disables.")
    parser.add_argument("--models", default="baseline,lstm,textcnn,distilbert",
                        help="Comma-separated: baseline,lstm,textcnn,distilbert")
    # Keras options
    parser.add_argument("--glove_path", default="", help="Optional path to GloVe txt (e.g., glove.6B.100d.txt).")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--vocab_size", type=int, default=30000)
    parser.add_argument("--emb_dim", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-3)
    # HF options
    parser.add_argument("--bert_epochs", type=int, default=3)
    parser.add_argument("--bert_batch_size", type=int, default=16)
    parser.add_argument("--bert_lr", type=float, default=2e-5)
    parser.add_argument("--bert_max_len", type=int, default=128)

    args = parser.parse_args()
    ensure_dir(args.out)
    seed_everything(args.seed)

    log(f"Loading dataset: {args.dataset} ...")
    X, y = load_data(args)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_split, stratify=y, random_state=args.seed
    )

    model_list = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    summary = {}

    if "baseline" in model_list:
        out_dir = os.path.join(args.out, "baseline")
        ensure_dir(out_dir)
        summary["baseline"] = run_baseline_tfidf_lr(X_train, y_train, X_val, y_val, out_dir)

    if "lstm" in model_list:
        out_dir = os.path.join(args.out, "lstm")
        ensure_dir(out_dir)
        summary["lstm"] = run_keras_model(
            "lstm",
            X_train, y_train, X_val, y_val, out_dir,
            vocab_size=args.vocab_size, seq_len=args.seq_len, emb_dim=args.emb_dim,
            glove_path=(args.glove_path if args.glove_path else None),
            batch_size=args.batch_size, epochs=args.epochs, lr=args.lr
        )

    if "textcnn" in model_list:
        out_dir = os.path.join(args.out, "textcnn")
        ensure_dir(out_dir)
        summary["textcnn"] = run_keras_model(
            "textcnn",
            X_train, y_train, X_val, y_val, out_dir,
            vocab_size=args.vocab_size, seq_len=args.seq_len, emb_dim=args.emb_dim,
            glove_path=(args.glove_path if args.glove_path else None),
            batch_size=args.batch_size, epochs=args.epochs, lr=args.lr
        )

    if "distilbert" in model_list:
        out_dir = os.path.join(args.out, "distilbert")
        ensure_dir(out_dir)
        summary["distilbert"] = run_distilbert(
            X_train, y_train, X_val, y_val, out_dir,
            epochs=args.bert_epochs, batch_size=args.bert_batch_size,
            lr=args.bert_lr, max_len=args.bert_max_len
        )

    # Select best by F1
    best_name, best_metrics = None, None
    for name, met in summary.items():
        if best_metrics is None or met.get("f1", 0.0) > best_metrics.get("f1", 0.0):
            best_name, best_metrics = name, met

    result = {
        "seed": args.seed,
        "val_split": args.val_split,
        "rows": int(len(X)),
        "models_trained": list(summary.keys()),
        "best_model": best_name,
        "best_f1": float(best_metrics.get("f1", np.nan)) if best_metrics else None,
        "all_metrics": summary,
    }
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    log(f"Done. Best model: {best_name} (F1={result['best_f1']:.3f})")
    log(f"Artifacts saved under: {args.out}")

if __name__ == "__main__":
    main()
