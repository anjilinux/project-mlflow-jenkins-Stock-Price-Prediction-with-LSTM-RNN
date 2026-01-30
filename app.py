from tensorflow.keras.models import load_model
import joblib
import numpy as np

MODEL_PATH = "models/lstm_model.keras"
SCALER_PATH = "scaler.pkl"

# Load model WITHOUT recompiling
model = load_model(MODEL_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)

print("✅ Model loaded successfully")

def predict(x):
    x = np.array(x).reshape(1, -1, 1)
    pred = model.predict(x)
    #return scaler.inverse_transform(pred)[0][0]
    print("🚀 App started successfully","###@@final-value is###$$$$$%",pred,scaler.inverse_transform(pred)[0][0])

if __name__ == "__main__":
    print("🚀 App started successfully","###@@final-value is###$$$$$%")
