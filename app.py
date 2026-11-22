
from pathlib import Path

from flask import Flask, render_template, request

from cases.adult_income.model import (
    initialize_model,
    prepare_user_input,
    get_form_options,
    explain_prediction
)

from cases.adult_income.visualizations import (
    generate_all_visualizations,
    generate_waterfall
)


app = Flask(__name__)

OUTPUT_DIR = (
    Path(app.root_path)
    / "static"
    / "generated"
    / "adult_income"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


print("Initialising Adult Income Classification model...")

components = initialize_model()

model = components["model"]
explainer = components["explainer"]
X_test = components["X_test"]

form_options = get_form_options()

print("Adult Income model initialised successfully.")

print("Generating global SHAP visualisations...")

visualizations = generate_all_visualizations(
    model=model,
    explainer=explainer,
    X_test=X_test,
    output_dir=OUTPUT_DIR
)

print("Adult Income application is ready.")


@app.route("/")
def index():
    """Display the main application homepage."""

    return render_template("index.html")


@app.route("/adult-income", methods=["GET"])
def adult_income():
    """Display the Adult Income prediction form."""

    return render_template(
        "adult_income.html",
        options=form_options,
        error=None
    )


@app.route("/adult-income/predict", methods=["POST"])
def adult_income_predict():
    """Generate an Adult Income prediction and SHAP explanation."""

    try:
        form_data = request.form.to_dict()

        print("Processing submitted Adult Income data...")

        sample = prepare_user_input(form_data)

        print("Generating prediction and SHAP explanation...")

        explanation = explain_prediction(
            model,
            explainer,
            sample
        )

        print("Generating individual SHAP waterfall plot...")

        generate_waterfall(
            explainer=explainer,
            sample=sample,
            output_dir=OUTPUT_DIR
        )

        print("Prediction and SHAP explanation generated successfully.")

        return render_template(
            "adult_income_results.html",
            explanation=explanation
        )

    except (ValueError, KeyError, TypeError) as error:

        app.logger.warning(
            "Prediction input error: %s",
            error
        )

        return render_template(
            "adult_income.html",
            options=form_options,
            error=f"Prediction input error: {error}"
        ), 400

    except Exception as error:

        app.logger.exception(
            "Unexpected error during Adult Income prediction."
        )

        return render_template(
            "adult_income.html",
            options=form_options,
            error="An unexpected error occurred while generating "
                  "the prediction. Please try again."
        ), 500


if __name__ == "__main__":
    app.run(
        debug=True,
        use_reloader=False
    )
