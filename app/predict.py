# app/predict.py

import joblib
import numpy as np

model = joblib.load("models/xgb_model.pkl")
scaler = joblib.load("models/scaler.pkl")

def predict_transaction(input_data, threshold=0.4):

    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    probability = float(model.predict_proba(input_scaled)[0][1])
    prediction = int(probability >= threshold)

    return prediction, probability