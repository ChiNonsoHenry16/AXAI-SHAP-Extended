
from pathlib import Path

from cases.california_housing.model import initialize_model

from cases.california_housing.visualizations import (
    create_explainer,
    generate_global_importance,
    generate_beeswarm,
    generate_waterfall,
    generate_dependence
)


def test_california_housing_visualizations():

    output_dir = (
        Path(__file__).resolve().parents[1]
        / "static"
        / "generated"
        / "california_housing"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Initialising California Housing model...")

    components = initialize_model()

    model = components["model"]
    X_test = components["X_test"]

    print("Creating SHAP explainer...")

    explainer = create_explainer(model)

    sample = X_test.iloc[[0]]

    print("Generating global feature importance...")

    global_importance = generate_global_importance(
        explainer,
        X_test,
        output_dir
    )

    print("Generating beeswarm plot...")

    beeswarm = generate_beeswarm(
        explainer,
        X_test,
        output_dir
    )

    print("Generating waterfall plot...")

    waterfall = generate_waterfall(
        explainer,
        sample,
        output_dir
    )

    print("Generating MedInc dependence plot...")

    dependence = generate_dependence(
        explainer,
        X_test,
        output_dir
    )

    outputs = [
        global_importance,
        beeswarm,
        waterfall,
        dependence
    ]

    for output in outputs:
        assert output.exists(), f"Missing output: {output}"
        assert output.stat().st_size > 0, f"Empty output: {output}"
        print(f"Verified: {output.name}")

    print("All four California Housing SHAP plots generated successfully.")


if __name__ == "__main__":
    test_california_housing_visualizations()
