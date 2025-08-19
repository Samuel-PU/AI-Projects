# waste_vgg16_transfer_learning.py
# =========================================
# Final Project: Classify Waste (Recyclable vs Organic) using VGG16 Transfer Learning
# Layout: data/TRAIN (split into train/val via validation_split), data/TEST for test
# Outputs: ./outputs_waste/<timestamp>/
# =========================================

import os
import json
import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow.keras.applications import vgg16
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# -----------------------------
# Config
# -----------------------------
SEED = 42
IMG_SIZE = (150, 150)   # Per brief
BATCH_SIZE = 32
EPOCHS_FE = 15          # Feature extraction
EPOCHS_FT = 15          # Fine-tuning
VAL_SPLIT = 0.2         # Split inside data/TRAIN into train/val
BASE_OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs_waste")
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, TIMESTAMP)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dataset roots (as per your current layout under data/)
DATA_DIRS = {
    "trainval": "data/TRAIN",    # single folder with subfolders per class
    "test":     "data/TEST",     # single folder with subfolders per class
    # "dataset": "data/DATASET"  # optional, unused here
}

# Reproducibility
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -----------------------------
# Helper: save a small text manifest
# -----------------------------
def save_manifest():
    manifest = {
        "timestamp": TIMESTAMP,
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs_feature_extraction": EPOCHS_FE,
        "epochs_fine_tuning": EPOCHS_FT,
        "data_dirs": DATA_DIRS,
        "val_split": VAL_SPLIT,
        "tf_version": tf.__version__,
    }
    with open(os.path.join(OUTPUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

# -----------------------------
# Task 1: Print TensorFlow version
# -----------------------------
print(f"[Task 1] TensorFlow version: {tf.__version__}")
save_manifest()

# -----------------------------
# Data Generators
# -----------------------------
# IMPORTANT: Use validation_split on both train and val datagens (same split value)
train_datagen = ImageDataGenerator(
    preprocessing_function=vgg16.preprocess_input,
    rotation_range=15,
    width_shift_range=0.08,
    height_shift_range=0.08,
    shear_range=0.08,
    zoom_range=0.10,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=VAL_SPLIT
)
val_datagen = ImageDataGenerator(
    preprocessing_function=vgg16.preprocess_input,
    validation_split=VAL_SPLIT
)
plain_datagen = ImageDataGenerator(preprocessing_function=vgg16.preprocess_input)

train_generator = train_datagen.flow_from_directory(
    DATA_DIRS["trainval"],
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    shuffle=True,
    seed=SEED
)

val_generator = val_datagen.flow_from_directory(
    DATA_DIRS["trainval"],
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

# -----------------------------
# Task 2: test_generator
# -----------------------------
test_generator = plain_datagen.flow_from_directory(
    DATA_DIRS["test"],
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)
print("[Task 2] test_generator created.")

# Save class index mapping
class_indices_path = os.path.join(OUTPUT_DIR, "class_indices.json")
with open(class_indices_path, "w") as f:
    json.dump(train_generator.class_indices, f, indent=2)
print(f"[Info] Saved class indices -> {class_indices_path}")

# -----------------------------
# Early data sanity check
# -----------------------------
def _count_images(gen): return getattr(gen, "samples", 0)
counts = {"train": _count_images(train_generator), "val": _count_images(val_generator), "test": _count_images(test_generator)}
print(f"[Data Check] Samples -> {counts}")
if counts["train"] == 0 or counts["val"] == 0:
    raise RuntimeError(
        "No images found for training/validation.\n"
        "Expected structure with files like:\n"
        "data/TRAIN/<class_name>/*.jpg\n"
        "data/TEST/<class_name>/*.jpg\n"
        f"Classes detected: {list(train_generator.class_indices.keys())}"
    )

# -----------------------------
# Build Models
# -----------------------------
def build_base_vgg(input_shape=(150, 150, 3)):
    return vgg16.VGG16(include_top=False, weights="imagenet", input_shape=input_shape)

def build_extract_features_model(input_shape=(150, 150, 3), dropout=0.3):
    base = build_base_vgg(input_shape)
    base.trainable = False  # Freeze for feature extraction
    inputs = layers.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs, outputs, name="extract_feat_model")
    return model

def compile_model(model, lr=1e-3):
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )

def plot_history(history, metric, title, filename):
    plt.figure()
    plt.plot(history.history[metric], label=f"train_{metric}")
    plt.plot(history.history[f"val_{metric}"], label=f"val_{metric}")
    plt.xlabel("Epoch")
    plt.ylabel(metric.capitalize())
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    out_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {out_path}")

def evaluate_and_print(model, test_gen, tag):
    loss, acc, auc = model.evaluate(test_gen, verbose=0)
    print(f"[{tag}] Test -> loss: {loss:.4f}, acc: {acc:.4f}, auc: {auc:.4f}")
    return {"loss": loss, "acc": acc, "auc": auc}

# -----------------------------
# Callbacks & paths
# -----------------------------
fe_ckpt = os.path.join(OUTPUT_DIR, "best_extract_feat.keras")
ft_ckpt = os.path.join(OUTPUT_DIR, "best_finetune.keras")

common_callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
]

fe_callbacks = [ModelCheckpoint(filepath=fe_ckpt, monitor="val_loss", save_best_only=True, verbose=1), *common_callbacks]
ft_callbacks = [ModelCheckpoint(filepath=ft_ckpt, monitor="val_loss", save_best_only=True, verbose=1), *common_callbacks]

# -----------------------------
# Train: Feature Extraction
# -----------------------------
extract_feat_model = build_extract_features_model(input_shape=(*IMG_SIZE, 3), dropout=0.3)
print("[Task 4] Extract Features Model Summary:")
extract_feat_model.summary()
compile_model(extract_feat_model, lr=1e-3)
print("[Task 5] Compiled extract features model.")

history_fe = extract_feat_model.fit(
    train_generator,
    epochs=EPOCHS_FE,
    validation_data=val_generator,
    callbacks=fe_callbacks,
    verbose=1
)

with open(os.path.join(OUTPUT_DIR, "history_extract_feat.json"), "w") as f:
    json.dump(history_fe.history, f, indent=2)

plot_history(history_fe, "accuracy", "Extract Features: Accuracy (Train vs Val)", "extract_features_accuracy.png")
plot_history(history_fe, "loss", "Extract Features: Loss (Train vs Val)", "extract_features_loss.png")

# -----------------------------
# Fine-Tuning (unfreeze block5_*)
# -----------------------------
# Load best FE weights if available
if os.path.exists(fe_ckpt):
    extract_feat_model.load_weights(fe_ckpt)
    print(f"[Info] Loaded best FE weights from {fe_ckpt}")

base_ft = build_base_vgg(input_shape=(*IMG_SIZE, 3))
base_ft.trainable = True
for layer in base_ft.layers:
    if "block5" not in layer.name:
        layer.trainable = False

inputs = layers.Input(shape=(*IMG_SIZE, 3))
x = base_ft(inputs, training=True)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
fine_tune_model = models.Model(inputs, outputs, name="fine_tune_model")

print("[Task 4] Fine-Tuned Model Summary:")
fine_tune_model.summary()
compile_model(fine_tune_model, lr=1e-5)

print("[Info] Starting fine-tuning...")
history_ft = fine_tune_model.fit(
    train_generator,
    epochs=EPOCHS_FT,
    validation_data=val_generator,
    callbacks=ft_callbacks,
    verbose=1
)

with open(os.path.join(OUTPUT_DIR, "history_finetune.json"), "w") as f:
    json.dump(history_ft.history, f, indent=2)

# -----------------------------
# Task 7 & 8: Curves
# -----------------------------
plot_history(history_ft, "loss", "Fine-Tuning: Loss (Train vs Val)", "finetune_loss.png")
plot_history(history_ft, "accuracy", "Fine-Tuning: Accuracy (Train vs Val)", "finetune_accuracy.png")

# -----------------------------
# Evaluate & Persist Metrics/Predictions
# -----------------------------
print("[Info] Evaluating on test set...")
metrics_fe = evaluate_and_print(extract_feat_model, test_generator, tag="ExtractFeatures")
metrics_ft = evaluate_and_print(fine_tune_model, test_generator, tag="FineTune")

with open(os.path.join(OUTPUT_DIR, "test_metrics.json"), "w") as f:
    json.dump({"extract_features": metrics_fe, "fine_tune": metrics_ft}, f, indent=2)

# Predictions CSV (fine-tuned)
y_prob = fine_tune_model.predict(test_generator, verbose=0).ravel()
y_pred = (y_prob >= 0.5).astype(int)
true_labels = test_generator.classes
filenames = test_generator.filenames
csv_path = os.path.join(OUTPUT_DIR, "test_predictions_finetune.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "true_label", "pred_label", "prob_recyclable"])
    for fn, t, p, pr in zip(filenames, true_labels, y_pred, y_prob):
        writer.writerow([fn, int(t), int(p), float(pr)])
print(f"[Saved] {csv_path}")

# -----------------------------
# Task 9 & 10: Example image visualisations
# -----------------------------
def load_image_for_plot(index, generator):
    filepath = generator.filepaths[index]
    img = tf.keras.utils.load_img(filepath, target_size=IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img)
    return img, arr, filepath

def predict_single(arr, model):
    x = np.expand_dims(arr, axis=0)
    x = vgg16.preprocess_input(x)
    prob = float(model.predict(x, verbose=0)[0][0])
    pred = int(prob >= 0.5)
    return pred, prob

idx_to_class = {v: k for k, v in test_generator.class_indices.items()}

index_to_plot = min(1, len(test_generator.filenames) - 1)  # safe guard
if len(test_generator.filenames) > 0:
    img, arr, path = load_image_for_plot(index_to_plot, test_generator)
    true_label_name = os.path.basename(os.path.dirname(path))

    # Task 9
    pred_fe, prob_fe = predict_single(arr, extract_feat_model)
    pred_fe_name = idx_to_class[pred_fe]
    plt.figure()
    plt.imshow(img)
    plt.title(f"[Extract Features]\nTrue: {true_label_name} | Pred: {pred_fe_name} | Prob(recyclable)= {prob_fe:.2f}")
    plt.axis("off")
    plt.savefig(os.path.join(OUTPUT_DIR, "task9_extract_features_img.png"), bbox_inches="tight")
    plt.close()

    # Task 10
    pred_ft, prob_ft = predict_single(arr, fine_tune_model)
    pred_ft_name = idx_to_class[pred_ft]
    plt.figure()
    plt.imshow(img)
    plt.title(f"[Fine-Tuned]\nTrue: {true_label_name} | Pred: {pred_ft_name} | Prob(recyclable)= {prob_ft:.2f}")
    plt.axis("off")
    plt.savefig(os.path.join(OUTPUT_DIR, "task10_finetune_img.png"), bbox_inches="tight")
    plt.close()
else:
    print("[Info] Skipping Task 9/10 example images because test set is empty.")

# -----------------------------
# Save trained models
# -----------------------------
extract_path = os.path.join(OUTPUT_DIR, "extract_features_model.keras")
finetune_path = os.path.join(OUTPUT_DIR, "fine_tuned_model.keras")
extract_feat_model.save(extract_path)
fine_tune_model.save(finetune_path)
print(f"[Saved] {extract_path}")
print(f"[Saved] {finetune_path}")
print(f"\nAll done. See outputs in: {OUTPUT_DIR}")
