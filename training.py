# ==========================================
# IMPORT LIBRARIES
# ==========================================

import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# ==========================================
# LOAD CLEAN DATASET
# ==========================================

df = pd.read_csv("datasets/yield_cleaned.csv")

# ==========================================
# FEATURES & TARGET
# ==========================================

X = df.drop("hg/ha_yield", axis=1)
y = df["hg/ha_yield"]

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# TRAIN MODEL
# ==========================================

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)

model.fit(X_train, y_train)

# ==========================================
# PREDICTION
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# EVALUATION
# ==========================================

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5

accuracy = r2 * 100

print(f"Accuracy : {accuracy:.2f}%")
print(f"R2 Score : {r2:.4f}")
print(f"MAE : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")


# ==========================================
# SAVE MODEL
# ==========================================

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/crop_yield_model.pkl")

print("Model Saved Successfully!")