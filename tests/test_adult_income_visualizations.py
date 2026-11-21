
from pathlib import Path

from cases.adult_income.model import initialize_model
from cases.adult_income.visualizations import (
    generate_all_visualizations
)


def main():
    print("=" * 60)
    print("ADULT INCOME - SHAP VISUALISATION TEST")
    print("=" * 60)

    print("\nInitialising Adult Income model...")

    components = initialize_model()

    output_dir = Path("static/generated/adult_income")

    print("\nGenerating visualisations...")

    results = generate_all_visualizations(
        model=components["model"],
        explainer=components["explainer"],
        X_test=components["X_test"],
        output_dir=output_dir
    )

    print("\nGenerated files:")

    for name, path in results.items():
        print(f"{name}: {path}")

        if not path.exists():
            raise FileNotFoundError(
                f"Expected visualisation not found: {path}"
            )

    print("\nAll four SHAP visualisations verified.")
    print("=" * 60)


if __name__ == "__main__":
    main()
