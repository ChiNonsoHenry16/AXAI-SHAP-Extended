
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import shap

from pathlib import Path


def generate_global_importance(
    shap_values,
    X_test,
    output_dir
):
    """Generate the SHAP global feature importance plot."""

    plt.figure()

    shap.summary_plot(
        shap_values,
        X_test,
        plot_type="bar",
        show=False
    )

    plt.title("SHAP Global Feature Importance")
    plt.tight_layout()

    output_path = Path(output_dir) / "shap_global_importance.png"

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return output_path


def generate_beeswarm(
    shap_values,
    X_test,
    output_dir
):
    """Generate the SHAP beeswarm plot."""

    plt.figure()

    shap.summary_plot(
        shap_values,
        X_test,
        show=False
    )

    plt.title("SHAP Feature Impact Summary")
    plt.tight_layout()

    output_path = Path(output_dir) / "shap_beeswarm.png"

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return output_path


def generate_waterfall(
    explainer,
    X_test,
    output_dir,
    sample_index=0
):
    """Generate a local SHAP waterfall plot."""

    sample = X_test.iloc[[sample_index]]

    sample_explanation = explainer(sample)

    plt.figure()

    shap.plots.waterfall(
        sample_explanation[0],
        show=False
    )

    plt.tight_layout()

    output_path = Path(output_dir) / "shap_waterfall.png"

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return output_path


def generate_dependence(
    shap_values,
    X_test,
    output_dir
):
    """Generate the Education-Num SHAP dependence plot."""

    plt.figure()

    shap.dependence_plot(
        "Education-Num",
        shap_values,
        X_test,
        show=False
    )

    plt.title("SHAP Dependence Plot: Education-Num")
    plt.tight_layout()

    output_path = (
        Path(output_dir) / "shap_dependence_education.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return output_path


def generate_all_visualizations(
    model,
    explainer,
    X_test,
    output_dir,
    sample_index=0
):
    """Generate all four original SHAP visualisations."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating SHAP values...")

    shap_values = explainer.shap_values(X_test)

    print("Generating global feature importance...")

    global_importance = generate_global_importance(
        shap_values,
        X_test,
        output_dir
    )

    print("Generating beeswarm plot...")

    beeswarm = generate_beeswarm(
        shap_values,
        X_test,
        output_dir
    )

    print("Generating waterfall plot...")

    waterfall = generate_waterfall(
        explainer,
        X_test,
        output_dir,
        sample_index
    )

    print("Generating Education-Num dependence plot...")

    dependence = generate_dependence(
        shap_values,
        X_test,
        output_dir
    )

    print("All SHAP visualisations generated successfully.")

    return {
        "global_importance": global_importance,
        "beeswarm": beeswarm,
        "waterfall": waterfall,
        "dependence": dependence
    }
