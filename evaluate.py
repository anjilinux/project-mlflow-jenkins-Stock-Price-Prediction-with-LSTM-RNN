from tensorflow.keras.models import load_model

MODEL_PATH = "lstm_model"
model = load_model(MODEL_PATH)
print("Model loaded successfully!")
