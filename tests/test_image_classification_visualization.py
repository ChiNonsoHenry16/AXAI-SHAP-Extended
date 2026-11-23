
from pathlib import Path

from cases.image_classification.model import get_sample_image
from cases.image_classification.visualizations import (
    generate_and_save_explanation
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "static"
    / "generated"
    / "image_classification"
)

OUTPUT_FILE = OUTPUT_DIR / "shap_imagenet_sample_01.png"


def test_generate_and_save_imagenet_visualization():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image = get_sample_image(1)

    explanation = generate_and_save_explanation(
        image,
        OUTPUT_FILE
    )

    assert explanation is not None
    assert OUTPUT_FILE.exists()
    assert OUTPUT_FILE.stat().st_size > 0
