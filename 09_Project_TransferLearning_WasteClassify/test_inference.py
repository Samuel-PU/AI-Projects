# test_inference.py
# =========================================
# CLI to run inference on a single image using a saved Keras model.
# Uses the same VGG16 preprocessing as training.
# =========================================

import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import vgg16

IMG_SIZE = (150, 150)

def predict_image(model_path: str, image_path: str):
    model = tf.keras.models.load_model(model_path)
    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(arr, axis=0)
    x = vgg16.preprocess_input(x)
    prob = float(model.predict(x, verbose=0)[0][0])
    label = "recyclable" if prob >= 0.5 else "organic"
    return label, prob

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to saved .keras model")
    parser.add_argument("--image", required=True, help="Path to image file to classify")
    args = parser.parse_args()

    label, prob = predict_image(args.model, args.image)
    print(f"Prediction: {label} | Prob(recyclable)={prob:.4f}")
