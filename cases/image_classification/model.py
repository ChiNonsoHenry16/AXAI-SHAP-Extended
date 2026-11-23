
from pathlib import Path
import json
import numpy as np
import shap
from tensorflow.keras.applications.resnet50 import (
    ResNet50,
    preprocess_input
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "datasets" / "imagenet50"

MODEL = None
IMAGE_DATA = None
IMAGE_LABELS = None
CLASS_NAMES = None


def load_model():
    """Load the pretrained ImageNet ResNet50 model."""
    global MODEL

    if MODEL is None:
        MODEL = ResNet50(weights="imagenet")

    return MODEL


def load_dataset():
    """Load the official SHAP ImageNet50 images and labels."""
    global IMAGE_DATA, IMAGE_LABELS

    if IMAGE_DATA is None:
        IMAGE_DATA, IMAGE_LABELS = shap.datasets.imagenet50()

    return IMAGE_DATA, IMAGE_LABELS


def load_class_names():
    """Load the 1,000 ImageNet class labels."""
    global CLASS_NAMES

    if CLASS_NAMES is None:
        url = (
            "https://storage.googleapis.com/"
            "download.tensorflow.org/data/imagenet_class_index.json"
        )

        class_index_path = shap.datasets.cache(url)

        with open(class_index_path, "r", encoding="utf-8") as file:
            class_index = json.load(file)

        CLASS_NAMES = [
            class_index[str(index)][1]
            for index in range(1000)
        ]

    return CLASS_NAMES


def preprocess_images(images):
    """Apply ResNet50 ImageNet preprocessing."""
    images = np.asarray(images, dtype=np.float32)
    return preprocess_input(images.copy())


def predict_probabilities(images):
    """Return ResNet50 ImageNet predictions."""
    model = load_model()
    processed_images = preprocess_images(images)

    return model.predict(processed_images, verbose=0)


def initialize_model():
    """Initialise ResNet50, ImageNet50 data and class labels."""
    model = load_model()
    images, labels = load_dataset()
    class_names = load_class_names()

    return {
        "model": model,
        "images": images,
        "labels": labels,
        "class_names": class_names
    }


def get_sample_image(index):
    """Retrieve one image from the SHAP ImageNet50 dataset."""
    images, _ = load_dataset()

    if index < 0 or index >= len(images):
        raise IndexError("Image index is outside the available dataset.")

    return images[index]


def predict_image(image):
    """Predict the top ImageNet class for an image."""
    probabilities = predict_probabilities(
        np.expand_dims(image, axis=0)
    )[0]

    top_index = int(np.argmax(probabilities))
    class_names = load_class_names()

    return {
        "class_index": top_index,
        "class_name": class_names[top_index],
        "probability": float(probabilities[top_index]),
        "probabilities": probabilities
    }
