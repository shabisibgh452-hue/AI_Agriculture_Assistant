import os
import sys
import joblib
import pandas as pd

# ==========================================
# IMPORT CONFIG
# ==========================================
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

# ==========================================
# LOAD MODEL
# ==========================================
model = joblib.load(FERTILIZER_MODEL)

# ==========================================
# LOAD ENCODERS & SCALER
# ==========================================
soil_encoder = joblib.load(SOIL_ENCODER)
crop_encoder = joblib.load(CROP_ENCODER)
fertilizer_encoder = joblib.load(FERTILIZER_ENCODER)
scaler = joblib.load(SCALER)

# ==========================================
# USER INPUT
# ==========================================
temperature = float(input("Temparature: "))
humidity = float(input("Humidity: "))
moisture = float(input("Moisture: "))
soil = input("Soil Type: ")
crop = input("Crop Type: ")
nitrogen = float(input("Nitrogen: "))
potassium = float(input("Potassium: "))
phosphorous = float(input("Phosphorous: "))

# ==========================================
# ENCODE INPUT
# ==========================================
soil = soil_encoder.transform([soil])[0]
crop = crop_encoder.transform([crop])[0]

# ==========================================
# CREATE DATAFRAME
# ==========================================
input_data = pd.DataFrame([[
    temperature,
    humidity,
    moisture,
    soil,
    crop,
    nitrogen,
    potassium,
    phosphorous
]], columns=[
    "Temparature",
    "Humidity",
    "Moisture",
    "Soil Type",
    "Crop Type",
    "Nitrogen",
    "Potassium",
    "Phosphorous"
])

# ==========================================
# SCALE INPUT
# ==========================================
input_scaled = scaler.transform(input_data)

# ==========================================
# PREDICT
# ==========================================
prediction = model.predict(input_scaled)

fertilizer = fertilizer_encoder.inverse_transform(prediction)

print("=" * 50)
print("🌱 Recommended Fertilizer:", fertilizer[0])
print("=" * 50)