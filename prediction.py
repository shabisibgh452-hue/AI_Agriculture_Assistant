# ==========================================
# IMPORT LIBRARIES
# ==========================================

import joblib

# ==========================================
# LOAD MODEL & ENCODERS
# ==========================================

model = joblib.load("models/crop_yield_model.pkl")
area_encoder = joblib.load("encoders/area_encoder.pkl")
item_encoder = joblib.load("encoders/item_encoder.pkl")

# ==========================================
# USER INPUT
# ==========================================

area = input("Enter Area: ")
item = input("Enter Crop Name: ")
year = int(input("Enter Year: "))
rainfall = float(input("Enter Average Rainfall (mm/year): "))
pesticides = float(input("Enter Pesticides (tonnes): "))
temperature = float(input("Enter Average Temperature: "))

# ==========================================
# ENCODE INPUT
# ==========================================

area = area_encoder.transform([area])[0]
item = item_encoder.transform([item])[0]

# ==========================================
# PREDICT
# ==========================================

prediction = model.predict([[
    area,
    item,
    year,
    rainfall,
    pesticides,
    temperature
]])

# ==========================================
# OUTPUT
# ==========================================

print("\n==============================")
print(f"Predicted Crop Yield: {prediction[0]:.2f} hg/ha")
print("==============================")