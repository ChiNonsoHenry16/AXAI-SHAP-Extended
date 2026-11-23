
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import shap

from pathlib import Path


def create_explainer(model):
    """Create a SHAP TreeExplainer for the trained model."""

    return shap.TreeExplainer(model)


def generate_global_importance(
    explainer,
    X_test,
    output_dir
):
    """Generate the global SHAP feature importance plot."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shap_values = explainer.shap_values(X_test)

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
    explainer,
    X_test,
    output_dir
):
    """Generate the SHAP beeswarm plot."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shap_values = explainer.shap_values(X_test)

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
    """Generate the local SHAP waterfall plot."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    explanation = explainer(sample)

    plt.figure()

    shap.plots.waterfall(
        explanation[0],
        max_display=8,
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
    explainer,
    X_test,
    output_dir,
    feature="MedInc"
):
    """Generate the SHAP dependence plot."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shap_values = explainer.shap_values(X_test)

    plt.figure()

    shap.dependence_plot(
        feature,
        shap_values,
        X_test,
        show=False
    )

    plt.title(f"SHAP Dependence Plot: {feature}")
    plt.tight_layout()

    output_path = output_dir / "shap_dependence_medinc.png"

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return output_path
