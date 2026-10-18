
from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os
import math

app = Flask(__name__)

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models",
    "predictive_maintenance_model.joblib"
)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)

FEATURES = list(model.feature_names_in_)

REQUIRED_FIELDS = [
    "type",
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear"
]


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "AI Predictive Maintenance API is running",
        "status": "success",
        "endpoints": {
            "home": "/",
            "prediction": "/predict"
        }
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({
                "error": "Please send input as a JSON object"
            }), 400

        missing_fields = [
            field for field in REQUIRED_FIELDS
            if field not in data
        ]

        if missing_fields:
            return jsonify({
                "error": "Missing required fields",
                "missing_fields": missing_fields
            }), 400

        product_type = str(data["type"]).strip().upper()

        if product_type not in ["L", "M", "H"]:
            return jsonify({
                "error": "Machine type must be L, M, or H"
            }), 400

        try:
            air_temperature = float(data["air_temperature"])
            process_temperature = float(data["process_temperature"])
            rotational_speed = float(data["rotational_speed"])
            torque = float(data["torque"])
            tool_wear = float(data["tool_wear"])

        except (ValueError, TypeError):
            return jsonify({
                "error": "All sensor values must be numeric"
            }), 400

        sensor_values = [
            air_temperature,
            process_temperature,
            rotational_speed,
            torque,
            tool_wear
        ]

        if not all(math.isfinite(value) for value in sensor_values):
            return jsonify({
                "error": "Sensor values must be finite numbers"
            }), 400

        if (
            air_temperature <= 0
            or process_temperature <= 0
            or rotational_speed <= 0
            or torque < 0
            or tool_wear < 0
        ):
            return jsonify({
                "error": "One or more sensor values are outside valid limits"
            }), 400

        input_data = pd.DataFrame([{
            "Air temperature [K]": air_temperature,
            "Process temperature [K]": process_temperature,
            "Rotational speed [rpm]": rotational_speed,
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear,
            "Type_L": int(product_type == "L"),
            "Type_M": int(product_type == "M")
        }])

        missing_features = [
            feature for feature in FEATURES
            if feature not in input_data.columns
        ]

        if missing_features:
            return jsonify({
                "error": "Input features do not match the trained model",
                "missing_features": missing_features,
                "model_features": FEATURES
            }), 500

        input_data = input_data.reindex(columns=FEATURES)

        prediction = int(model.predict(input_data)[0])

        result = {
            "machine_failure": prediction,
            "status": (
                "Failure risk detected"
                if prediction == 1
                else "No failure predicted"
            ),
            "machine_type": product_type,
            "prediction_note": (
                "This is a model prediction, not a guarantee."
            )
        }

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            classes = list(model.classes_)

            if 1 in classes:
                failure_index = classes.index(1)
                result["failure_probability_percent"] = round(
                    float(probabilities[failure_index]) * 100, 2
                )

        return jsonify(result), 200

    except Exception as e:
        app.logger.exception("Prediction failed")

        return jsonify({
            "error": "Prediction could not be completed",
            "details": str(e)
        }), 500


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
