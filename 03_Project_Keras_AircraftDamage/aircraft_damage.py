# Classification and Captioning: Aircraft Damage (VGG16 + optional BLIP)
# ---------------------------------------------------------------------
# This script trains a binary classifier (dent vs crack) using transfer learning

import os
import sys
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless plots
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import requests
from io import BytesIO

# --------------------------
# Reproducibility & settings
# --------------------------
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# --------------------------
# Paths & dataset structure
# --------------------------
script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
train_dir = os.path.join(script_dir, "train")
valid_dir = os.path.join(script_dir, "valid")
test_dir  = os.path.join(script_dir, "test")

print("Using dataset paths:")
print(f"  train: {train_dir}")
print(f"  valid: {valid_dir}")
print(f"  test : {test_dir}")

for path in (train_dir, valid_dir, test_dir):
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"Directory not found: {path}\n"
            "Expected structure:\n"
            "  train/\n"
            "    crack/\n"
            "    dent/\n"
            "  valid/\n"
            "    crack/\n"
            "    dent/\n"
            "  test/\n"
            "    crack/\n"
            "    dent/\n"
        )

# --------------------------
# Image params & generators
# --------------------------
img_size = (224, 224)  # VGG16 default
batch_size = 16

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen  = ImageDataGenerator(rescale=1./255)

print("\nCreating data generators...")
train_gen = train_datagen.flow_from_directory(
    train_dir, target_size=img_size, batch_size=batch_size, class_mode="binary", seed=seed_value
)
valid_gen = valid_datagen.flow_from_directory(
    valid_dir, target_size=img_size, batch_size=batch_size, class_mode="binary", seed=seed_value, shuffle=False
)
test_gen = test_datagen.flow_from_directory(
    test_dir, target_size=img_size, batch_size=batch_size, class_mode="binary", seed=seed_value, shuffle=False
)

print(f"Class indices: {train_gen.class_indices}")

# --------------------------
# Model: VGG16 transfer learn
# --------------------------
print("\nBuilding model...")
base = VGG16(weights="imagenet", include_top=False, input_shape=(img_size[0], img_size[1], 3))
for layer in base.layers:
    layer.trainable = False

model = Sequential([
    base,
    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid"),
])

model.compile(optimizer=Adam(1e-4), loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# --------------------------
# Train
# --------------------------
print("\nTraining...")
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
]

history = model.fit(
    train_gen,
    epochs=20,
    validation_data=valid_gen,
    callbacks=callbacks,
    verbose=1
)

train_history = history.history

# --------------------------
# Plots
# --------------------------
print("\nSaving training curves...")
plt.figure(figsize=(5,5))
plt.plot(train_history["accuracy"], label="Train")
plt.plot(train_history["val_accuracy"], label="Valid")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "accuracy_curve.png"))
plt.close()

plt.figure(figsize=(5,5))
plt.plot(train_history["loss"], label="Train")
plt.plot(train_history["val_loss"], label="Valid")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "loss_curve.png"))
plt.close()

# --------------------------
# Evaluate
# --------------------------
print("\nEvaluating on test set...")
test_gen.reset()
test_loss, test_acc = model.evaluate(test_gen, verbose=1)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Loss    : {test_loss:.4f}")

# --------------------------
# Visualise one prediction
# --------------------------
def plot_image_with_title(img, true_label, pred_label, class_names, out_path):
    plt.figure(figsize=(6,6))
    plt.imshow(img)
    plt.title(f"True: {class_names[int(true_label)]}\nPred: {class_names[int(pred_label)]}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")

print("\nCreating prediction visualisation...")
test_gen.reset()
test_images, test_labels = next(test_gen)
pred_probs = model.predict(test_images, verbose=0).ravel()
pred_classes = (pred_probs > 0.5).astype(int)

class_names = {v: k for k, v in test_gen.class_indices.items()}
idx = min(1, len(test_images)-1)
plot_image_with_title(
    test_images[idx], test_labels[idx], pred_classes[idx], class_names,
    os.path.join(script_dir, "prediction_visualization.png")
)

# --------------------------
# PART 2: Optional captioning
# --------------------------
print("\nCaptioning a sample image (optional)...")
image_path = os.path.join(script_dir, "aircraft_damage.jpg")

def try_download_sample(path):
    urls = [
        "https://images.unsplash.com/photo-1550751827-4bd374c3f58b?auto=format&fit=crop&w=800&q=80",
        "https://images.unsplash.com/photo-1581091226033-d5c48150dbaa?auto=format&fit=crop&w=800&q=80",
        "https://images.unsplash.com/photo-1581092580497-e0d23cbdf1dc?auto=format&fit=crop&w=800&q=80",
    ]
    headers = {"User-Agent": "Mozilla/5.0"}
    for u in urls:
        try:
            r = requests.get(u, headers=headers, timeout=10)
            r.raise_for_status()
            img = Image.open(BytesIO(r.content)).convert("RGB")
            img.save(path)
            print(f"Sample image saved to: {path}")
            return True
        except Exception as e:
            print(f"  - Failed to fetch {u}: {e}")
    return False

if not os.path.exists(image_path):
    if not try_download_sample(image_path):
        # Fallback to test batch image
        test_gen.reset()
        x, _ = next(test_gen)
        fallback = (x[0] * 255).astype(np.uint8)
        Image.fromarray(fallback).save(image_path)
        print(f"Used fallback test image: {image_path}")
else:
    print(f"Using existing sample image: {image_path}")

# Try BLIP captioning if dependencies are available
caption = "An aircraft image."
try:
    import importlib
    has_torch = importlib.util.find_spec("torch") is not None
    has_transformers = importlib.util.find_spec("transformers") is not None

    if has_torch and has_transformers:
        from transformers import pipeline
        print("Loading BLIP captioning pipeline...")
        cap_pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
        caption = cap_pipe(Image.open(image_path))[0]["generated_text"]
    else:
        if not has_torch:
            print("Skipping BLIP: PyTorch not installed.")
        if not has_transformers:
            print("Skipping BLIP: transformers not installed.")
        caption = "An aircraft with visible structural features."
except Exception as e:
    print(f"Captioning error, using placeholder: {e}")
    caption = "An aircraft with visible structural features."

# --------------------------
# Save summary outputs
# --------------------------
out_txt = os.path.join(script_dir, "project_outputs.txt")
with open(out_txt, "w", encoding="utf-8") as f:
    f.write("PART 1: CLASSIFICATION OUTPUTS\n")
    f.write("="*60 + "\n")
    f.write(f"Test Accuracy: {test_acc:.4f}\n")
    f.write(f"Test Loss    : {test_loss:.4f}\n\n")
    f.write("PART 2: CAPTIONING\n")
    f.write("="*60 + "\n")
    f.write(f"Caption: {caption}\n")

print("\nArtifacts saved:")
print(" - accuracy_curve.png")
print(" - loss_curve.png")
print(" - prediction_visualization.png")
print(f" - {out_txt}")
