import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder

# Load Dataset
df = pd.read_csv("datasets/yield_df.csv")

print("Original Shape:", df.shape)

# Remove unnecessary column
if "Unnamed: 0" in df.columns:
    df.drop("Unnamed: 0", axis=1, inplace=True)

# Remove missing values
df.dropna(inplace=True)

# Encode categorical columns
area_encoder = LabelEncoder()
item_encoder = LabelEncoder()

df["Area"] = area_encoder.fit_transform(df["Area"])
df["Item"] = item_encoder.fit_transform(df["Item"])

# Create folders
os.makedirs("encoders", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Save encoders
joblib.dump(area_encoder, "encoders/area_encoder.pkl")
joblib.dump(item_encoder, "encoders/item_encoder.pkl")

# Save cleaned dataset
df.to_csv("datasets/yield_cleaned.csv", index=False)

print("Preprocessing Completed Successfully!")
print(df.head())