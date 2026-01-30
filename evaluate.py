from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
import numpy as np

model = load_model("models/lstm_model.h5")

def evaluate(X_test, y_test):
    preds = model.predict(X_test)
    return mean_squared_error(y_test, preds)
