
import shap
import xgboost as xgb
import pandas as pd

from sklearn.model_selection import train_test_split


def load_dataset():
    """Load the SHAP Adult Income dataset."""

    X, y = shap.datasets.adult()
    y = y.astype(int)

    return X, y


def prepare_data():
    """Load and split the dataset."""

    X, y = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    return X, y, X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """Train the original XGBoost classification model."""

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)

    return model


def create_explainer(model):
    """Create the SHAP TreeExplainer."""

    return shap.TreeExplainer(model)


def explain_prediction(model, explainer, sample):
    """Generate a prediction and SHAP explanation."""

    prediction = model.predict(sample)[0]
    probability = model.predict_proba(sample)[0][1]

    shap_values = explainer.shap_values(sample)

    feature_contributions = []

    for feature, value in zip(sample.columns, shap_values[0]):
        feature_contributions.append({
            "feature": feature,
            "shap_value": float(value),
            "effect": (
                "increases prediction"
                if value > 0
                else "decreases prediction"
            )
        })

    feature_contributions.sort(
        key=lambda item: abs(item["shap_value"]),
        reverse=True
    )

    return {
        "prediction": int(prediction),
        "probability": float(probability),
        "shap_values": feature_contributions
    }


def prepare_user_input(form_data):
    """Convert form values into the original model input format."""

    X_display, _ = shap.datasets.adult(display=True)

    categorical_features = [
        "Workclass",
        "Marital Status",
        "Occupation",
        "Relationship",
        "Race",
        "Sex",
        "Country"
    ]

    input_data = {
        "Age": float(form_data["Age"]),
        "Education-Num": float(form_data["Education-Num"]),
        "Capital Gain": float(form_data["Capital Gain"]),
        "Capital Loss": float(form_data["Capital Loss"]),
        "Hours per week": float(form_data["Hours per week"])
    }

    for feature in categorical_features:
        categories = X_display[feature].cat.categories

        category_lookup = {
            str(category).strip(): category
            for category in categories
        }

        selected_value = form_data[feature].strip()

        if selected_value not in category_lookup:
            raise ValueError(
                f"Invalid value for {feature}: {selected_value}"
            )

        original_category = category_lookup[selected_value]

        input_data[feature] = categories.get_loc(original_category)

    feature_order = [
        "Age",
        "Workclass",
        "Education-Num",
        "Marital Status",
        "Occupation",
        "Relationship",
        "Race",
        "Sex",
        "Capital Gain",
        "Capital Loss",
        "Hours per week",
        "Country"
    ]

    return pd.DataFrame([input_data], columns=feature_order)


def get_form_options():
    """Return human-readable categorical values."""

    X_display, _ = shap.datasets.adult(display=True)

    categorical_features = [
        "Workclass",
        "Marital Status",
        "Occupation",
        "Relationship",
        "Race",
        "Sex",
        "Country"
    ]

    options = {}

    for feature in categorical_features:
        options[feature] = [
            str(value).strip()
            for value in X_display[feature].cat.categories
        ]

    return options


def initialize_model():
    """Load data, train the model, and create the explainer."""

    X, y, X_train, X_test, y_train, y_test = prepare_data()

    model = train_model(X_train, y_train)
    explainer = create_explainer(model)

    return {
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "model": model,
        "explainer": explainer
    }
