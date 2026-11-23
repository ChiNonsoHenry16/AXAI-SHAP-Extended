
import numpy as np
from PIL import Image


def validate_image(image):
    """Validate an image from the ImageNet sample dataset."""
    image = np.asarray(image)

    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("Expected an RGB image with three channels.")

    if image.shape[0] != 224 or image.shape[1] != 224:
        raise ValueError("Expected an image with dimensions 224 x 224.")

    return image


def prepare_image_for_display(image):
    """Convert an image into a display-compatible RGB array."""
    image = validate_image(image)
    image = np.clip(image, 0, 255).astype(np.uint8)

    return Image.fromarray(image)


def prepare_image_batch(images):
    """Validate and prepare a batch of RGB images."""
    return np.asarray([
        validate_image(image)
        for image in images
    ])
