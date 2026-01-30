import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

WINDOW_SIZE = 60

def load_assets():
    model = load_model("models/lstm_model.h5")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler

def prepare_input(csv_path):
    df = pd.read_csv(csv_path)
    close_prices = df[['Close']].values

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close_prices)

    joblib.dump(scaler, "models/scaler.pkl")

    last_window = scaled[-WINDOW_SIZE:]
    return np.expand_dims(last_window, axis=0), scaler

def predict_next_price(csv_path):
    X_input, scaler = prepare_input(csv_path)
    model, _ = load_assets()

    scaled_prediction = model.predict(X_input)
    prediction = scaler.inverse_transform(scaled_prediction)

    return float(prediction[0][0])

if __name__ == "__main__":
    price = predict_next_price("data/raw/stock_prices.csv")
    print(f"📈 Predicted next closing price: {price}")
