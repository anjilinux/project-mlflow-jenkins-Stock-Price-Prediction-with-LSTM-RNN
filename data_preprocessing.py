import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def prepare_data(file_path, window_size=60):
    df = pd.read_csv(file_path)
    close_prices = df[['Close']].values

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close_prices)

    X, y = [], []
    for i in range(window_size, len(scaled)):
        X.append(scaled[i-window_size:i])
        y.append(scaled[i])

    return np.array(X), np.array(y), scaler
