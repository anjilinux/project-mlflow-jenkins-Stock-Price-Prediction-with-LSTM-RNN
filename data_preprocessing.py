import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

WINDOW_SIZE = 60

def prepare_data(
    raw_csv_path,
    processed_csv_path="processed.csv"
):
    df = pd.read_csv(raw_csv_path)

    close_df = df[['Close']].copy()

    scaler = MinMaxScaler()
    close_df['Close_scaled'] = scaler.fit_transform(close_df[['Close']])

    # ✅ SAFE directory creation
    dir_name = os.path.dirname(processed_csv_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    close_df.to_csv(processed_csv_path, index=False)

    values = close_df['Close_scaled'].values

    X, y = [], []
    for i in range(WINDOW_SIZE, len(values)):
        X.append(values[i-WINDOW_SIZE:i])
        y.append(values[i])

    X = np.array(X).reshape(-1, WINDOW_SIZE, 1)
    y = np.array(y)

    return X, y, scaler
