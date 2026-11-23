
import numpy as np
import shap

from cases.image_classification.model import (
    predict_probabilities,
    load_class_names
)


def create_explainer(image):
    """Create the official SHAP image masker and Partition explainer."""
    masker = shap.maskers.Image(
        "inpaint_telea",
        image.shape
    )

    explainer = shap.Explainer(
        predict_probabilities,
        masker,
        output_names=load_class_names()
    )

    return explainer


def generate_shap_explanation(image):
    """Generate a SHAP explanation for one ImageNet image."""
    explainer = create_explainer(image)

    explanation = explainer(
        np.expand_dims(image, axis=0),
        max_evals=500,
        batch_size=50,
        outputs=shap.Explanation.argsort.flip[:1]
    )

    return explanation



def save_shap_visualization(explanation, output_path):
    """Inspect and save the official SHAP image plot."""
    import matplotlib
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import numpy as np

    plt.close("all")

    #shap.image_plot(
        #explanation,
        #show=False
    #)

    shap.image_plot(
    explanation,
    pixel_values=explanation.data.astype(np.uint8),
    show=False
    )

    for figure_number in plt.get_fignums():
        figure = plt.figure(figure_number)

        original_artist = figure.axes[0].images[0]
        displayed_image = np.asarray(original_artist.get_array())

        original_image = np.asarray(explanation.data[0])

        print("\nOriginal image comparison:")
        print("Displayed image shape:", displayed_image.shape)
        print("Explanation image shape:", original_image.shape)

        print(
            "Displayed image range:",
            displayed_image.min(),
            displayed_image.max()
        )

        print(
            "Explanation image range:",
            original_image.min(),
            original_image.max()
        )

        print(
            "Displayed image mean:",
            displayed_image.mean()
        )

        print(
            "Explanation image mean:",
            original_image.mean()
        )

        print(
            "Displayed image matches normalized explanation:",
            np.allclose(
                displayed_image,
                original_image / 255.0
            )
        )

        figure.savefig(
            output_path,
            bbox_inches="tight",
            dpi=150
        )

    plt.close("all")



def generate_and_save_explanation(image, output_path):
    """Generate and save a SHAP image explanation."""
    explanation = generate_shap_explanation(image)

    save_shap_visualization(
        explanation,
        output_path
    )

    return explanation
