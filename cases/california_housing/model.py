import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import train_test_split


def load_dataset():
    """Load the California Housing dataset."""

    dataset = fetch_california_housing(as_frame=True)

    X = dataset.data
    y = dataset.target

    return X, y


def prepare_data(X, y):
    """Split the dataset into training and testing subsets."""

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )


def train_model(X_train, y_train):
    """Train the HistGradientBoostingRegressor model."""

    model = HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.1,
        max_leaf_nodes=31,
        l2_regularization=0.1,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate regression model performance."""

    predictions = model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
        "R2": r2_score(y_test, predictions)
    }

    return metrics


def initialize_model():
    """Load data, train the model and evaluate performance."""

    X, y = load_dataset()

    X_train, X_test, y_train, y_test = prepare_data(X, y)

    model = train_model(X_train, y_train)

    metrics = evaluate_model(
        model,
        X_test,
        y_test
    )

    return {
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "model": model,
        "metrics": metrics,
        "feature_names": X.columns.tolist()
    }


def prepare_user_input(form_data):
    """Convert submitted housing features into a model input."""

    feature_names = [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude"
    ]

    values = {}

    for feature in feature_names:
        value = float(form_data[feature])

        if not np.isfinite(value):
            raise ValueError(
                f"{feature} must be a valid finite number."
            )

        values[feature] = value

    import pandas as pd

    return pd.DataFrame(
        [values],
        columns=feature_names
    )


def predict_house_value(model, sample):
    """Predict the median house value."""

    prediction = model.predict(sample)[0]

    return {
        "predicted_value": float(prediction),
        "predicted_value_usd": float(prediction * 100000)
    }
