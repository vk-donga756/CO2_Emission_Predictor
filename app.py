from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

# ===== MODEL SELECTION =====
# 🔧 TO SWITCH MODELS: Uncomment ONE line below, save file, Flask will auto-reload
# MODEL_CHOICE = "lr_model.joblib"      # Linear Regression (Fast, less accurate)
MODEL_CHOICE = "rf_model.joblib"      # Random Forest (BEST - Balanced speed/accuracy) ✅
# MODEL_CHOICE = "gbr_model.joblib"     # Gradient Boosting (Slower, very accurate)
# MODEL_CHOICE = "co2_model.joblib"     # Default (currently Random Forest)

# Load the selected model
model_path = os.path.join(os.path.dirname(__file__), MODEL_CHOICE)
try:
    model = joblib.load(model_path)
    print("=" * 70)
    print(f"✅ MODEL LOADED SUCCESSFULLY")
    print("=" * 70)
    print(f"   File:       {MODEL_CHOICE}")
    print(f"   Model Type: {type(model.named_steps['model']).__name__}")
    print(f"   Dataset:    Fuel_Consumption_2000-2022.csv")
    print("=" * 70)
except FileNotFoundError:
    print("=" * 70)
    print(f"❌ ERROR: {MODEL_CHOICE} not found!")
    print("=" * 70)
    print("   Please run the notebook to train and save models first:")
    print("   1. Open 'ML lab project (2).ipynb'")
    print("   2. Run main() function")
    print("   3. Restart this Flask app")
    print("=" * 70)
    model = None
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

@app.route("/")
def home():
    if model is None:
        return "❌ CO2 Prediction API - Model not loaded!", 500
    
    model_type = type(model.named_steps['model']).__name__
    html = f"""
    <html>
    <head><title>CO2 Prediction API</title></head>
    <body style="font-family: Arial; padding: 40px; background: #1a1a1a; color: #fff;">
        <h1>✅ CO2 Prediction API is Live!</h1>
        <hr>
        <h2>Current Model Configuration:</h2>
        <ul>
            <li><strong>Model File:</strong> {MODEL_CHOICE}</li>
            <li><strong>Model Type:</strong> {model_type}</li>
            <li><strong>Dataset:</strong> Fuel_Consumption_2000-2022.csv</li>
        </ul>
        <hr>
        <h2>How to Switch Models:</h2>
        <ol>
            <li>Open <code>app.py</code></li>
            <li>Change <code>MODEL_CHOICE</code> variable (lines 11-14)</li>
            <li>Save the file (Flask will auto-reload)</li>
            <li>Refresh this page to confirm</li>
        </ol>
        <hr>
        <p><a href="/model-info" style="color: #4CAF50;">View Model Info (JSON)</a></p>
    </body>
    </html>
    """
    return html

@app.route("/model-info", methods=["GET"])
def model_info():
    """Return information about the loaded model"""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    return jsonify({
        "model_file": MODEL_CHOICE,
        "model_type": type(model.named_steps['model']).__name__,
        "features": [
            "Engine size (L)", 
            "Cylinders", 
            "Combined (L/100 km)", 
            "Fuel type", 
            "Vehicle class", 
            "Transmission"
        ],
        "target": "CO2 emissions (g/km)"
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({"error": "Model not loaded. Please check server logs."}), 500
        
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
        
        # Log the prediction for debugging
        print(f"\n🎯 Prediction made:")
        print(f"   Model: {type(model.named_steps['model']).__name__}")
        print(f"   Input: Engine={data['engine_size']}L, Cyl={data['cylinders']}, Fuel={data['fuel_consumption']}L/100km")
        print(f"   Result: {prediction:.2f} g/km")
        
        return jsonify({
            "prediction": round(float(prediction), 2),
            "model_used": MODEL_CHOICE.replace('.joblib', ''),
            "model_type": type(model.named_steps['model']).__name__
        })

    except Exception as e:
        print("❌ Error during prediction:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Enable debug mode so Flask auto-reloads when you change MODEL_CHOICE
    app.run(host="0.0.0.0", port=5000, debug=True)
