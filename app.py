from pathlib import Path

from flask import Flask, jsonify, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

PROJECT_DIR = Path(__file__).resolve().parent
model = joblib.load(PROJECT_DIR / "best_model.pkl")
scaler = joblib.load(PROJECT_DIR / "scaler.pkl")
selector = joblib.load(PROJECT_DIR / "selectkbest.pkl")
FEATURE_ORDER = selector.feature_names_in_.tolist()

CATEGORY_OPTIONS = {
    "Gender": ("Female", "Male"),
    "Partner": ("No", "Yes"),
    "Dependents": ("No", "Yes"),
    "PhoneService": ("No", "Yes"),
    "MultipleLines": ("No phone service", "No", "Yes"),
    "InternetService": ("DSL", "Fiber optic", "No"),
    "OnlineSecurity": ("No internet service", "No", "Yes"),
    "OnlineBackup": ("No internet service", "No", "Yes"),
    "DeviceProtection": ("No internet service", "No", "Yes"),
    "TechSupport": ("No internet service", "No", "Yes"),
    "StreamingTV": ("No internet service", "No", "Yes"),
    "StreamingMovies": ("No internet service", "No", "Yes"),
    "Contract": ("Month-to-month", "One year", "Two year"),
    "PaperlessBilling": ("No", "Yes"),
    "PaymentMethod": (
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ),
}


def numeric_value(data, field):
    try:
        value = float(data[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"'{field}' must be a valid number.") from exc

    if not np.isfinite(value):
        raise ValueError(f"'{field}' must be a finite number.")
    return value


def category_value(data, field):
    value = data.get(field)
    if value not in CATEGORY_OPTIONS[field]:
        options = ", ".join(CATEGORY_OPTIONS[field])
        raise ValueError(f"'{field}' must be one of: {options}.")
    return value


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/result")
def result():
    return render_template("result.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Send the customer details as a JSON object."
        }), 400

    try:
        # -----------------------------
        # 1. Validate categorical values
        # -----------------------------
        categories = {
            field: category_value(data, field)
            for field in CATEGORY_OPTIONS
        }

        # -----------------------------
        # 2. Encode categorical values
        # -----------------------------
        encoded = {
            "SeniorCitizen": numeric_value(data, "SeniorCitizen"),
            "tenure": numeric_value(data, "tenure"),
            "MonthlyCharges": numeric_value(data, "MonthlyCharges"),
            "TotalCharges": numeric_value(data, "TotalCharges"),

            "gender_Male": int(categories["Gender"] == "Male"),
            "Partner_Yes": int(categories["Partner"] == "Yes"),
            "Dependents_Yes": int(categories["Dependents"] == "Yes"),
            "PhoneService_Yes": int(categories["PhoneService"] == "Yes"),

            "MultipleLines_No phone service": int(
                categories["MultipleLines"] == "No phone service"
            ),
            "MultipleLines_Yes": int(
                categories["MultipleLines"] == "Yes"
            ),

            "InternetService_Fiber optic": int(
                categories["InternetService"] == "Fiber optic"
            ),
            "InternetService_No": int(
                categories["InternetService"] == "No"
            ),

            "OnlineSecurity_No internet service": int(
                categories["OnlineSecurity"] == "No internet service"
            ),
            "OnlineSecurity_Yes": int(
                categories["OnlineSecurity"] == "Yes"
            ),

            "OnlineBackup_No internet service": int(
                categories["OnlineBackup"] == "No internet service"
            ),
            "OnlineBackup_Yes": int(
                categories["OnlineBackup"] == "Yes"
            ),

            "DeviceProtection_No internet service": int(
                categories["DeviceProtection"] == "No internet service"
            ),
            "DeviceProtection_Yes": int(
                categories["DeviceProtection"] == "Yes"
            ),

            "TechSupport_No internet service": int(
                categories["TechSupport"] == "No internet service"
            ),
            "TechSupport_Yes": int(
                categories["TechSupport"] == "Yes"
            ),

            "StreamingTV_No internet service": int(
                categories["StreamingTV"] == "No internet service"
            ),
            "StreamingTV_Yes": int(
                categories["StreamingTV"] == "Yes"
            ),

            "StreamingMovies_No internet service": int(
                categories["StreamingMovies"] == "No internet service"
            ),
            "StreamingMovies_Yes": int(
                categories["StreamingMovies"] == "Yes"
            ),

            "Contract_One year": int(
                categories["Contract"] == "One year"
            ),
            "Contract_Two year": int(
                categories["Contract"] == "Two year"
            ),

            "PaperlessBilling_Yes": int(
                categories["PaperlessBilling"] == "Yes"
            ),

            "PaymentMethod_Credit card (automatic)": int(
                categories["PaymentMethod"] == "Credit card (automatic)"
            ),
            "PaymentMethod_Electronic check": int(
                categories["PaymentMethod"] == "Electronic check"
            ),
            "PaymentMethod_Mailed check": int(
                categories["PaymentMethod"] == "Mailed check"
            ),
        }

        # -----------------------------
        # 3. Arrange features
        # -----------------------------
        full_features = np.array(
            [[encoded[field] for field in FEATURE_ORDER]],
            dtype=float
        )

        # -----------------------------
        # 4. SelectKBest
        # -----------------------------
        selected_features = full_features[:, selector.get_support()]

        # -----------------------------
        # 5. StandardScaler
        # -----------------------------
        scaled_features = scaler.transform(selected_features)

        # -----------------------------
        # 6. Prediction
        # -----------------------------
        prediction = int(model.predict(scaled_features)[0])

        # -----------------------------
        # 7. Probabilities
        # -----------------------------
        probabilities = model.predict_proba(scaled_features)[0]

        stay_probability = float(probabilities[0] * 100)
        churn_probability = float(probabilities[1] * 100)

        # -----------------------------
        # 8. Risk Level
        # -----------------------------
        if churn_probability >= 70:
            risk_level = "High Risk"

            recommendation = (
                "Customer has a high probability of churn. "
                "Contact the customer immediately and provide "
                "retention offers, discounts, personalized support, "
                "or service improvements."
            )

        elif churn_probability >= 40:
            risk_level = "Medium Risk"

            recommendation = (
                "Customer has a moderate probability of churn. "
                "Monitor the customer closely and provide "
                "personalized offers, better service support, "
                "or loyalty benefits."
            )

        else:
            risk_level = "Low Risk"

            recommendation = (
                "Customer has a low probability of churn. "
                "Continue regular engagement and maintain "
                "good customer service."
            )

        # -----------------------------
        # 9. Final Result
        # -----------------------------
        result = (
            "Customer will Churn"
            if prediction == 1
            else "Customer will Stay"
        )

        # -----------------------------
        # 10. Return JSON
        # -----------------------------
        return jsonify({
            "prediction": prediction,
            "result": result,
            "churn_probability": round(churn_probability, 2),
            "stay_probability": round(stay_probability, 2),
            "risk_level": risk_level,
            "recommendation": recommendation
        })
    except ValueError as exc:
        return jsonify({
            "error": str(exc)
        }), 400

    except Exception as exc:
        return jsonify({
            "error": str(exc)
        }), 500