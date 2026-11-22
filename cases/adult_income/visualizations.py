
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

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()

    shap.summary_plot(
        shap_values,
        X_test,
        plot_type="bar",
        show=False
    )

    plt.title("SHAP Global Feature Importance")
    plt.tight_layout()

    output_path = output_dir / "shap_global_importance.png"

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

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()

    shap.summary_plot(
        shap_values,
        X_test,
        show=False
    )

    plt.title("SHAP Feature Impact Summary")
    plt.tight_layout()

    output_path = output_dir / "shap_beeswarm.png"

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return output_path


def generate_waterfall(
    explainer,
    sample,
    output_dir
):
    """Generate a local SHAP waterfall plot for an observation."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_explanation = explainer(sample)

    plt.figure()

    shap.plots.waterfall(
        sample_explanation[0],
        show=False
    )

    plt.tight_layout()

    output_path = output_dir / "shap_waterfall.png"

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

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()

    shap.dependence_plot(
        "Education-Num",
        shap_values,
        X_test,
        show=False
    )

    plt.title("SHAP Dependence Plot: Education-Num")
    plt.tight_layout()

    output_path = output_dir / "shap_dependence_education.png"

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
    sample=None
):
    """
    Generate global SHAP plots and a local waterfall explanation.

    If no sample is supplied, the first test observation is used
    for the initial waterfall plot during application startup.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if sample is None:
        sample = X_test.iloc[[0]]

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
        sample,
        output_dir
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
