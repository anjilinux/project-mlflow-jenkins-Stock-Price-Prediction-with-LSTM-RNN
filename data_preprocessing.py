import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

WINDOW_SIZE = 60

def prepare_data(raw_csv_path, processed_csv_path="processed.csv"):
    df = pd.read_csv(raw_csv_path)

    # Keep only required column
    close_prices = df[['Close']].copy()

    scaler = MinMaxScaler()
    close_prices['Close_scaled'] = scaler.fit_transform(close_prices[['Close']])

    # Save processed data
    os.makedirs(os.path.dirname(processed_csv_path), exist_ok=True)
    close_prices.to_csv(processed_csv_path, index=False)

    # Create sequences
    scaled_values = close_prices['Close_scaled'].values

    X, y = [], []
    for i in range(WINDOW_SIZE, len(scaled_values)):
        X.append(scaled_values[i-WINDOW_SIZE:i])
        y.append(scaled_values[i])

    X = np.array(X).reshape(-1, WINDOW_SIZE, 1)
    y = np.array(y)

    return X, y, scaler
