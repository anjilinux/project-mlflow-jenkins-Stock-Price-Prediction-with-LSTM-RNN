import os
import mlflow
import mlflow.tensorflow
import joblib

from data_preprocessing import prepare_data
from model import build_lstm

# Paths
RAW_CSV = "stock_prices.csv"
PROCESSED_CSV = "processed.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "lstm_model")  # SavedModel format
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# MLflow experiment
mlflow.set_experiment("Stock_Price_LSTM")

# Prepare data
X, y, scaler = prepare_data(
    raw_csv_path=RAW_CSV,
    processed_csv_path=PROCESSED_CSV,
    window_size=5
)

# Save scaler
joblib.dump(scaler, SCALER_PATH)

# Start MLflow run
with mlflow.start_run():
    # Build model
    model = build_lstm((X.shape[1], 1))
    
    # Train
    history = model.fit(X, y, epochs=10, batch_size=32)

    # Log parameters and metrics
    mlflow.log_param("epochs", 10)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("window_size", X.shape[1])
    mlflow.log_metric("loss", history.history["loss"][-1])

    # Save model in **TensorFlow SavedModel format**
    model.save(MODEL_PATH)

    # Log model to MLflow
    mlflow.tensorflow.log_model(model, "model")

    # Log scaler and processed CSV
    mlflow.log_artifact(SCALER_PATH)
    mlflow.log_artifact(PROCESSED_CSV)

print(f"Model saved at {MODEL_PATH}")
print(f"Scaler saved at {SCALER_PATH}")
