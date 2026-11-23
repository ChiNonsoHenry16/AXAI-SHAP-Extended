
from cases.image_classification.model import (
    load_model,
    load_dataset,
    load_class_names
)


def test_imagenet_model_and_dataset():
    model = load_model()
    images, labels = load_dataset()
    class_names = load_class_names()

    assert model.name == "resnet50"
    assert len(images) == 50
    assert len(class_names) == 1000
    assert images.shape[1:] == (224, 224, 3)
    assert len(labels) == 50
