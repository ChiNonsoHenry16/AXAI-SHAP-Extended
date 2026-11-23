from cases.image_classification.model import get_sample_image
from cases.image_classification.visualizations import create_explainer


def test_partition_explainer_initialization():
    image = get_sample_image(1)

    explainer = create_explainer(image)

    assert explainer is not None
    assert explainer.__class__.__name__ == "PartitionExplainer"
    assert explainer.masker is not None
