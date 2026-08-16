import os
import sys
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ==========================================
# IMPORT CONFIG
# ==========================================
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

# ==========================================
# LOAD DATASET
# ==========================================
df = pd.read_csv(FERTILIZER_DATASET)

# ==========================================
# REMOVE MISSING VALUES
# ==========================================
df.dropna(inplace=True)

# ==========================================
# ENCODE CATEGORICAL COLUMNS
# ==========================================
soil_encoder = LabelEncoder()
crop_encoder = LabelEncoder()
fertilizer_encoder = LabelEncoder()

df["Soil Type"] = soil_encoder.fit_transform(df["Soil Type"])
df["Crop Type"] = crop_encoder.fit_transform(df["Crop Type"])
df["Fertilizer Name"] = fertilizer_encoder.fit_transform(df["Fertilizer Name"])

# ==========================================
# FEATURES & TARGET
# ==========================================
X = df.drop(columns=["Fertilizer Name"])
y = df["Fertilizer Name"]

# ==========================================
# FEATURE SCALING
# ==========================================
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ==========================================
# SAVE ENCODERS
# ==========================================
joblib.dump(soil_encoder, SOIL_ENCODER)
joblib.dump(crop_encoder, CROP_ENCODER)
joblib.dump(fertilizer_encoder, FERTILIZER_ENCODER)
joblib.dump(scaler, SCALER)

print("=" * 50)
print("Fertilizer Preprocessing Completed Successfully!")
print("=" * 50)