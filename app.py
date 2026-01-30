from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model("models/lstm_model.h5")

@app.route("/predict", methods=["POST"])
def predict():
    data = np.array(request.json["data"])
    pred = model.predict(data).tolist()
    return jsonify({"prediction": pred})

if __name__ == "__main__":
    app.run(port=5000)
