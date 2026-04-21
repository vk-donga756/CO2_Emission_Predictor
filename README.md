# 🌍 CO2 Emission Predictor

A machine learning web application that predicts vehicle CO2 emissions based on engine specifications and fuel consumption patterns. Built with Flask and scikit-learn, trained on Canadian fuel consumption data (2000-2022).

## 🎯 Features

- **Multiple ML Models**: Choose between Linear Regression, Random Forest, or Gradient Boosting
- **RESTful API**: Simple JSON-based prediction endpoint
- **Interactive Interface**: Web UI for easy predictions
- **Real-time Switching**: Hot-swap between models without restarting
- **Production Ready**: Configured for deployment with Gunicorn and Procfile

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- pip

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/CO2_Emission_Predictor.git
cd CO2_Emission_Predictor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train the models (if not already present):
   - Open and run the Jupyter notebook: `ML lab project (2).ipynb`
   - This will generate the model files (.joblib)

4. Run the application:
```bash
python app.py
```

The API will be available at `http://localhost:5000`

## 📊 API Usage

### Make a Prediction
**Endpoint**: `POST /predict`

**Request Body**:
```json
{
  "engine_size": 3.5,
  "cylinders": 6,
  "fuel_consumption": 11.2,
  "fuel_type": "Regular gasoline",
  "vehicle_class": "SUV - Small",
  "transmission": "Automatic"
}
```

**Response**:
```json
{
  "prediction": 265.43,
  "model_used": "rf_model",
  "model_type": "RandomForestRegressor"
}
```

### Model Information
**Endpoint**: `GET /model-info`

Returns details about the currently loaded model.

## 🔧 Switching Models

To use a different model, edit `app.py` line 13:

```python
# MODEL_CHOICE = "lr_model.joblib"      # Linear Regression
MODEL_CHOICE = "rf_model.joblib"      # Random Forest (Default)
# MODEL_CHOICE = "gbr_model.joblib"     # Gradient Boosting
```

Flask will automatically reload with the new model.

## 📈 Model Performance

- **Linear Regression**: Fast, baseline accuracy
- **Random Forest**: Best balance of speed and accuracy (Recommended)
- **Gradient Boosting**: Highest accuracy, slower predictions

## 🗂️ Project Structure

```
CO2_Emission_Predictor/
├── app.py                          # Flask API server
├── requirements.txt                # Python dependencies
├── Procfile                        # Deployment configuration
├── Fuel_Consumption_2000-2022.csv # Training dataset
├── ML lab project (2).ipynb       # Model training notebook
├── lr_model.joblib                # Linear Regression model
├── rf_model.joblib                # Random Forest model
├── gbr_model.joblib               # Gradient Boosting model
└── co2_model.joblib               # Default model
```

## 🌐 Deployment

The application includes a `Procfile` for easy deployment to platforms like Heroku:

```bash
web: gunicorn app:app
```

## 📊 Dataset

Trained on the Canadian Vehicle Fuel Consumption dataset (2000-2022) containing:
- Engine size
- Number of cylinders
- Fuel consumption (city/highway/combined)
- Fuel type
- Vehicle class
- Transmission type
- CO2 emissions

## 🛠️ Technologies Used

- **Backend**: Flask, Flask-CORS
- **ML**: scikit-learn, pandas, numpy
- **Deployment**: Gunicorn
- **Model Persistence**: joblib


## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements!

