from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

# Use relative path for model (works on Render)
model_path = os.path.join(os.path.dirname(__file__), "co2_model.joblib")
model = joblib.load(model_path)

@app.route("/")
def home():
    return "✅ CO2 Prediction API is live!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Build dataframe directly from input (string features intact)
        df = pd.DataFrame([[
            data["engine_size"],
            data["cylinders"],
            data["fuel_consumption"],
            data["fuel_type"],
            data["vehicle_class"],
            data["transmission"]
        ]], columns=[
            "Engine size (L)",
            "Cylinders",
            "Combined (L/100 km)",
            "Fuel type",
            "Vehicle class",
            "Transmission"
        ])

        # Predict
        prediction = model.predict(df)[0]
        return jsonify({"prediction": round(float(prediction), 2)})

    except Exception as e:
        print("❌ Error during prediction:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
