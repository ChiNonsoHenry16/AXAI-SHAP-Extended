import numpy as np

from cases.image_classification.model import (
    get_sample_image,
    predict_image
)


def test_imagenet_image_prediction():
    image = get_sample_image(1)

    prediction = predict_image(image)

    assert isinstance(prediction["class_name"], str)
    assert prediction["class_name"]
    assert 0 <= prediction["class_index"] < 1000
    assert 0 <= prediction["probability"] <= 1
    assert len(prediction["probabilities"]) == 1000
    assert np.isfinite(prediction["probabilities"]).all()
