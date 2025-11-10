from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS  # 👈 important for allowing HTML to call API

app = Flask(__name__)
CORS(app)

# Load trained model
model = joblib.load("C:\\Users\\Harsh Kumar\\Desktop\\5th sem\\ML\\lab+lab project\\co2_model.joblib")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    
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
    
    prediction = model.predict(df)[0]
    return jsonify({"prediction": round(float(prediction), 2)})

if __name__ == "__main__":
    app.run(debug=True)
