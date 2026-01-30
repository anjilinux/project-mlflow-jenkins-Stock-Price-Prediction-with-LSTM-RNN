from tensorflow.keras.models import load_model
import os

MODEL_PATH = "models/lstm_model.h5"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train.py first.")

model = load_model(MODEL_PATH)
print("Model loaded successfully!")
