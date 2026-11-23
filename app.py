
from pathlib import Path

from flask import Flask, render_template, request

from cases.adult_income.model import (
    initialize_model as initialize_adult_model,
    prepare_user_input as prepare_adult_user_input,
    get_form_options,
    explain_prediction as explain_adult_prediction
)

from cases.adult_income.visualizations import (
    generate_all_visualizations as generate_adult_visualizations,
    generate_waterfall as generate_adult_waterfall
)

from cases.california_housing.model import (
    initialize_model as initialize_california_model,
    prepare_user_input as prepare_california_user_input,
    predict_house_value
)

from cases.california_housing.visualizations import (
    create_explainer as create_california_explainer,
    generate_waterfall as generate_california_waterfall
)


app = Flask(__name__)

OUTPUT_DIR = Path(app.root_path) / "static" / "generated"

ADULT_OUTPUT_DIR = OUTPUT_DIR / "adult_income"
CALIFORNIA_OUTPUT_DIR = OUTPUT_DIR / "california_housing"

ADULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CALIFORNIA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Adult Income model initialisation

print("Initialising Adult Income Classification model...")

adult_components = initialize_adult_model()

adult_model = adult_components["model"]
adult_explainer = adult_components["explainer"]
adult_X_test = adult_components["X_test"]

adult_form_options = get_form_options()

print("Adult Income model initialised successfully.")

print("Generating Adult Income SHAP visualisations...")

adult_visualizations = generate_adult_visualizations(
    model=adult_model,
    explainer=adult_explainer,
    X_test=adult_X_test,
    output_dir=ADULT_OUTPUT_DIR
)

print("Adult Income application components ready.")


# California Housing model initialisation

print("Initialising California Housing Regression model...")

california_components = initialize_california_model()

california_model = california_components["model"]
california_X_test = california_components["X_test"]
california_metrics = california_components["metrics"]

print("California Housing model initialised successfully.")

print("Creating California Housing SHAP explainer...")

california_explainer = create_california_explainer(
    california_model
)

print("California Housing application components ready.")


# Homepage

@app.route("/")
def index():
    """Display the main application homepage."""

    return render_template("index.html")


# Adult Income routes

@app.route("/adult-income", methods=["GET"])
def adult_income():
    """Display the Adult Income prediction form."""

    return render_template(
        "adult_income.html",
        options=adult_form_options,
        error=None
    )


@app.route("/adult-income/predict", methods=["POST"])
def adult_income_predict():
    """Generate an Adult Income prediction and SHAP explanation."""

    try:
        form_data = request.form.to_dict()

        sample = prepare_adult_user_input(form_data)

        explanation = explain_adult_prediction(
            adult_model,
            adult_explainer,
            sample
        )

        print("Generating individual Adult Income waterfall plot...")

        generate_adult_waterfall(
            explainer=adult_explainer,
            sample=sample,
            output_dir=ADULT_OUTPUT_DIR
        )

        print("Adult Income prediction generated successfully.")

        return render_template(
            "adult_income_results.html",
            explanation=explanation
        )

    except (ValueError, KeyError, TypeError) as error:

        app.logger.warning(
            "Adult Income prediction input error: %s",
            error
        )

        return render_template(
            "adult_income.html",
            options=adult_form_options,
            error=f"Prediction input error: {error}"
        ), 400

    except Exception:

        app.logger.exception(
            "Unexpected error during Adult Income prediction."
        )

        return render_template(
            "adult_income.html",
            options=adult_form_options,
            error="An unexpected error occurred while generating "
                  "the prediction. Please try again."
        ), 500


# California Housing routes

@app.route("/california-housing", methods=["GET"])
def california_housing():
    """Display the California Housing prediction form."""

    return render_template(
        "california_housing.html",
        error=None
    )


@app.route("/california-housing/predict", methods=["POST"])
def california_housing_predict():
    """Generate a California Housing prediction and SHAP explanation."""

    try:
        form_data = request.form.to_dict()

        print("Processing California Housing input...")

        sample = prepare_california_user_input(form_data)

        print("Generating California Housing prediction...")

        explanation = predict_house_value(
            california_model,
            sample
        )

        print("Generating individual California Housing SHAP values...")

        shap_explanation = california_explainer(sample)

        shap_values = shap_explanation.values[0]

        feature_contributions = []

        for feature, value in zip(sample.columns, shap_values):

            contribution = float(value)

            if contribution > 0:
                effect = "Increases prediction"
            elif contribution < 0:
                effect = "Decreases prediction"
            else:
                effect = "No effect"

            feature_contributions.append({
                "feature": feature,
                "shap_value": contribution,
                "effect": effect
            })

        feature_contributions.sort(
            key=lambda item: abs(item["shap_value"]),
            reverse=True
        )

        explanation["shap_values"] = feature_contributions

        print("Generating individual California Housing waterfall plot...")

        generate_california_waterfall(
            explainer=california_explainer,
            sample=sample,
            output_dir=CALIFORNIA_OUTPUT_DIR
        )

        print("California Housing prediction generated successfully.")

        return render_template(
            "california_housing_results.html",
            explanation=explanation,
            metrics=california_metrics
        )

    except (ValueError, KeyError, TypeError) as error:

        app.logger.warning(
            "California Housing prediction input error: %s",
            error
        )

        return render_template(
            "california_housing.html",
            error=f"Prediction input error: {error}"
        ), 400

    except Exception:

        app.logger.exception(
            "Unexpected error during California Housing prediction."
        )

        return render_template(
            "california_housing.html",
            error="An unexpected error occurred while generating "
                  "the prediction. Please try again."
        ), 500


# Application startup

if __name__ == "__main__":

    print("\nRegistered Flask routes:")

    print(app.url_map)

    print("\nAXAI-SHAP Extended application is ready.")

    app.run(
        debug=True,
        use_reloader=False
    )
