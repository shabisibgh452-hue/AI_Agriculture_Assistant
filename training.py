import os
import sys
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ==========================================
# IMPORT CONFIG
# ==========================================
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

# ==========================================
# LOAD DATASET
# ==========================================
df = pd.read_csv(FERTILIZER_DATASET)
df.dropna(inplace=True)

# ==========================================
# LOAD ENCODERS & SCALER
# ==========================================
soil_encoder = joblib.load(SOIL_ENCODER)
crop_encoder = joblib.load(CROP_ENCODER)
fertilizer_encoder = joblib.load(FERTILIZER_ENCODER)
scaler = joblib.load(SCALER)

# ==========================================
# ENCODE CATEGORICAL COLUMNS
# ==========================================
df["Soil Type"] = soil_encoder.transform(df["Soil Type"])
df["Crop Type"] = crop_encoder.transform(df["Crop Type"])
df["Fertilizer Name"] = fertilizer_encoder.transform(df["Fertilizer Name"])

# ==========================================
# FEATURES & TARGET
# ==========================================
X = df.drop("Fertilizer Name", axis=1)
y = df["Fertilizer Name"]

# ==========================================
# FEATURE SCALING
# ==========================================
X = scaler.transform(X)

# ==========================================
# TRAIN TEST SPLIT
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================================
# TRAIN MODEL
# ==========================================
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# ==========================================
# EVALUATION
# ==========================================
train_accuracy = accuracy_score(y_train, model.predict(X_train))
test_accuracy = accuracy_score(y_test, model.predict(X_test))
cv_accuracy = cross_val_score(model, X, y, cv=5).mean()

print("=" * 50)
print(f"Training Accuracy : {train_accuracy*100:.2f}%")
print(f"Testing Accuracy  : {test_accuracy*100:.2f}%")
print(f"Cross Validation  : {cv_accuracy*100:.2f}%")
print("=" * 50)

# ==========================================
# SAVE MODEL
# ==========================================
joblib.dump(model, FERTILIZER_MODEL)

print("✅ Fertilizer Model Saved Successfully!")