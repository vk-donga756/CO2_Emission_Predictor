# --- train_model.py ---
# This script is run ONCE on your local computer.
# Its only job is to load your local data and save the trained model.

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import joblib # <-- Import joblib to save the model

print("Script started...")

# --- 1. Load Your LOCAL Data ---
# ! IMPORTANT: Update this path to point to your local CSV file
LOCAL_DATA_PATH = "C:\\Users\\Harsh Kumar\\Desktop\\5th sem\\ML\\lab+lab project\\my2026-fuel-consumption-ratings.csv"
# LOCAL_DATA_PATH = "FuelConsumption.csv" # Or this, if it's in the same folder

try:
    data = pd.read_csv(LOCAL_DATA_PATH)
    print("Local data loaded successfully.")
except FileNotFoundError:
    print(f"ERROR: File not found at {LOCAL_DATA_PATH}")
    print("Please update the LOCAL_DATA_PATH variable in this script.")
    exit()

# --- 2. Define Features and Target ---
# ! IMPORTANT: Update these names to match your CSV columns
numeric_features = ['Engine size (L)', 'Cylinders', 'Combined (L/100 km)']
categorical_features = ['Fuel type', 'Vehicle class', 'Transmission']
target = 'CO2 emissions (g/km)'

# Check if columns exist
all_features = numeric_features + categorical_features + [target]
missing_cols = [col for col in all_features if col not in data.columns]
if missing_cols:
    print(f"ERROR: The following columns are missing from your CSV: {missing_cols}")
    print("Please update the feature/target names in this script.")
    exit()

features = numeric_features + categorical_features
X = data[features]
y = data[target]

# --- 3. Create Preprocessing Pipeline ---
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# --- 4. Create and Train Random Forest Model ---
print("Training Random Forest model...")
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
])

# Train the model on the ENTIRE dataset
rf_pipeline.fit(X, y)
print("Model training complete.")

# --- 5. Save the Trained Model to a File ---
# This is the most important step!
# It creates a new file named 'co2_model.joblib'
# This file contains your entire trained pipeline.
model_filename = 'co2_model.joblib'
joblib.dump(rf_pipeline, model_filename)

print("-" * 50)
print(f"SUCCESS! Model saved to '{model_filename}'")
print("You can now upload this file to a public URL.")
print("-" * 50)