import numpy as np

from cases.image_classification.model import get_sample_image
from cases.image_classification.visualizations import (
    generate_shap_explanation
)


def test_generate_imagenet_shap_explanation():
    image = get_sample_image(1)

    explanation = generate_shap_explanation(image)

    assert explanation is not None
    assert explanation.values.shape[0] == 1
    assert np.isfinite(explanation.values).all()
