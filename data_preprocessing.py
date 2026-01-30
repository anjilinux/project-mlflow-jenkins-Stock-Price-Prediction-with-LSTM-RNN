import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

DEFAULT_WINDOW_SIZE = 5

def prepare_data(
    raw_csv_path,
    processed_csv_path="processed.csv",
    window_size=DEFAULT_WINDOW_SIZE
):
    df = pd.read_csv(raw_csv_path)

    # Jenkins-safe: dynamic adjustment
    window_size = min(window_size, max(len(df)-1, 1))

    close_df = df[['Close']].copy()
    scaler = MinMaxScaler()
    close_df['Close_scaled'] = scaler.fit_transform(close_df[['Close']])

    dir_name = os.path.dirname(processed_csv_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    close_df.to_csv(processed_csv_path, index=False)

    values = close_df['Close_scaled'].values
    X, y = [], []
    for i in range(window_size, len(values)):
        X.append(values[i-window_size:i])
        y.append(values[i])

    X = np.array(X).reshape(-1, window_size, 1)
    y = np.array(y)

    if X.shape[0] == 0:
        raise RuntimeError("No training samples generated after preprocessing.")

    return X, y, scaler
