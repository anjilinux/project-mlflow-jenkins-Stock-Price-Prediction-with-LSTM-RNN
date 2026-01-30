import mlflow
import mlflow.tensorflow
import joblib

from data_preprocessing import prepare_data
from model import build_lstm

mlflow.set_experiment("Stock_Price_LSTM")

X, y, scaler = prepare_data(
    raw_csv_path="stock_prices.csv",
    processed_csv_path="processed.csv",
    window_size=5
)

joblib.dump(scaler, "scaler.pkl")

with mlflow.start_run():

    model = build_lstm((X.shape[1], 1))
    history = model.fit(X, y, epochs=10, batch_size=32)

    mlflow.log_param("epochs", 10)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("window_size", X.shape[1])
    mlflow.log_metric("loss", history.history["loss"][-1])

    model.save("models/lstm_model.h5")

    mlflow.tensorflow.log_model(model, "model")
    mlflow.log_artifact("scaler.pkl")
    mlflow.log_artifact("processed.csv")
