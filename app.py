from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import joblib
import numpy as np

# ------------------------
# Config
# ------------------------
MODEL_PATH = "models/lstm_model.keras"
SCALER_PATH = "scaler.pkl"
PORT = 5000

# ------------------------
# App
# ------------------------
app = Flask(__name__)

# ------------------------
# Load model & scaler once
# ------------------------
model = load_model(MODEL_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)

print("✅ Model and scaler loaded successfully")

# ------------------------
# Routes
# ------------------------

@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "message": "Stock Price LSTM API is running 🚀"
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Expecting: { "values": [v1, v2, v3, ...] }
        values = data.get("values")

        if not values:
            return jsonify({"error": "Missing 'values' in request"}), 400

        # Convert to model input shape (1, timesteps, 1)
        x = np.array(values).reshape(1, -1, 1)

        # Predict
        scaled_pred = model.predict(x)
        prediction = scaler.inverse_transform(scaled_pred)[0][0]

        return jsonify({
            "prediction": float(prediction)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------------
# Run
# ------------------------
if __name__ == "__main__":
    print(f"🚀 Flask app running on port {PORT}")
    app.run(host="0.0.0.0", port=PORT)





# from tensorflow.keras.models import load_model
# import joblib
# import numpy as np

# MODEL_PATH = "models/lstm_model.keras"
# SCALER_PATH = "scaler.pkl"

# # Load model WITHOUT recompiling
# model = load_model(MODEL_PATH, compile=False)
# scaler = joblib.load(SCALER_PATH)

# print("✅ Model loaded successfully")

# def predict(x):
#     x = np.array(x).reshape(1, -1, 1)
#     pred = model.predict(x)
#     #return scaler.inverse_transform(pred)[0][0]
#     print("🚀 App started successfully","###@@final-value is###$$$$$%",pred,scaler.inverse_transform(pred)[0][0])

# if __name__ == "__main__":
#     print("🚀 App started successfully","###@@final-value is###$$$$$%")
