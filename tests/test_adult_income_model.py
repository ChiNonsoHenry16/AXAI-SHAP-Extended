
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from cases.adult_income.model import (
    prepare_data,
    train_model,
    create_explainer,
    explain_prediction
)


def main():
    print("=" * 60)
    print("ADULT INCOME CLASSIFICATION - MODEL TEST")
    print("=" * 60)

    print("\nLoading Adult Income dataset...")

    X, y, X_train, X_test, y_train, y_test = prepare_data()

    print(f"Dataset shape: {X.shape}")
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")

    print("\nTraining XGBoost model...")

    model = train_model(X_train, y_train)

    print("Creating SHAP TreeExplainer...")

    explainer = create_explainer(model)

    print("Model and explainer initialized successfully.")

    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)

    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_probability)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\n" + "=" * 60)
    print("SINGLE OBSERVATION PREDICTION")
    print("=" * 60)

    sample = X_test.iloc[[0]]

    explanation = explain_prediction(
        model,
        explainer,
        sample
    )

    if explanation["prediction"] == 1:
        predicted_income = "> $50K"
    else:
        predicted_income = "<= $50K"

    print(f"Predicted income: {predicted_income}")

    print(
        f"Probability of income > $50K: "
        f"{explanation['probability']:.4f}"
    )

    print("\n" + "=" * 60)
    print("SHAP FEATURE CONTRIBUTIONS")
    print("=" * 60)

    for item in explanation["shap_values"]:
        print(
            f"{item['feature']}: "
            f"{item['shap_value']:.6f} "
            f"({item['effect']})"
        )

    print("\n" + "=" * 60)
    print("MODEL TEST COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
