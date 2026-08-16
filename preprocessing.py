import pandas as pd

# Load Dataset
df = pd.read_csv("../datasets/Crop_recommendation.csv")

# Remove Duplicates
df = df.drop_duplicates()

# Fill Missing Values
df = df.fillna(df.median(numeric_only=True))

# Save Cleaned Dataset
df.to_csv("../datasets/cleaned_crop_data.csv", index=False)

print("Preprocessing Completed Successfully!")