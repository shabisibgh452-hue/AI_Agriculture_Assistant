import joblib
import pandas as pd

# Load Model
model = joblib.load("../models/best_crop_model.pkl")

# Load Encoder
encoder = joblib.load("../encoders/crop_encoder.pkl")

# Sample Input
sample = pd.DataFrame({
    "N": [90],
    "P": [42],
    "K": [43],
    "temperature": [20.87],
    "humidity": [82.00],
    "ph": [6.50],
    "rainfall": [202.93]
})

# Predict
prediction = model.predict(sample)

# Decode Prediction
crop = encoder.inverse_transform(prediction)

print("Recommended Crop:", crop[0])