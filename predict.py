import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

WINDOW_SIZE = 60
MODEL_PATH = "models/lstm_model.keras"
SCALER_PATH = "models/scaler.pkl"

def load_assets():
    """Load trained model and scaler"""
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

def prepare_input(csv_path, scaler):
    """Prepare last WINDOW_SIZE data points for prediction"""
    df = pd.read_csv(csv_path)
    close_prices = df[['Close']].values

    scaled = scaler.transform(close_prices)  # ✅ use trained scaler
    last_window = scaled[-WINDOW_SIZE:]
    return np.expand_dims(last_window, axis=0)

def predict_next_price(csv_path):
    model, scaler = load_assets()
    X_input = prepare_input(csv_path, scaler)

    scaled_prediction = model.predict(X_input)
    prediction = scaler.inverse_transform(scaled_prediction)
    return float(prediction[0][0])

if __name__ == "__main__":
    price = predict_next_price("data/raw/stock_prices.csv")
    print(f"📈 Predicted next closing price: {price}")
