import mlflow
import mlflow.tensorflow
import joblib

from data_preprocessing import prepare_data
from model import build_lstm

# Set MLflow experiment
mlflow.set_experiment("Stock_Price_LSTM")

# Prepare data
X, y, scaler = prepare_data("data/raw/stock_prices.csv")

# Save scaler for inference
joblib.dump(scaler, "models/scaler.pkl")

with mlflow.start_run():

    # Build & train model
    model = build_lstm((X.shape[1], X.shape[2]))
    history = model.fit(X, y, epochs=10, batch_size=32)

    # Log params & metrics
    mlflow.log_param("epochs", 10)
    mlflow.log_param("batch_size", 32)
    mlflow.log_metric("loss", history.history["loss"][-1])

    # Save model
    model.save("models/lstm_model.h5")

    # Log artifacts
    mlflow.tensorflow.log_model(model, artifact_path="model")
    mlflow.log_artifact("models/scaler.pkl")
