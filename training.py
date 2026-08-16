import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# ==========================================
# ROOT DIRECTORY
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_DIR = os.path.join(BASE_DIR, "models")
ENCODER_DIR = os.path.join(BASE_DIR, "encoders")
DATASET_PATH = os.path.join(
    BASE_DIR,
    "datasets",
    "cleaned_crop_data.csv"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ENCODER_DIR, exist_ok=True)


# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv(DATASET_PATH)

print("Dataset Loaded Successfully!")
print("Dataset Shape:", df.shape)


# ==========================================
# FEATURES & TARGET
# ==========================================

X = df.drop("label", axis=1)
y = df["label"]


# ==========================================
# ENCODE TARGET
# ==========================================

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

print("\nCrop Classes:")
print(encoder.classes_)

print("\nNumber of Classes:", len(encoder.classes_))


# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)


# ==========================================
# MODELS
# ==========================================

models = {

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        random_state=42,
        eval_metric="mlogloss"
    ),

    "LightGBM": LGBMClassifier(
        n_estimators=300,
        random_state=42
    ),

    "Deep Learning": MLPClassifier(
        hidden_layer_sizes=(100, 50),
        max_iter=500,
        random_state=42
    )
}


# ==========================================
# TRAIN MODELS
# ==========================================

best_model = None
best_accuracy = 0

print("\n==============================")
print("MODEL ACCURACIES")
print("==============================")

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    ) * 100

    print(f"{name}: {accuracy:.2f}%")

    if accuracy > best_accuracy:

        best_accuracy = accuracy
        best_model = model


# ==========================================
# SAVE MODEL
# ==========================================

model_path = os.path.join(
    MODEL_DIR,
    "best_crop_model.pkl"
)

encoder_path = os.path.join(
    ENCODER_DIR,
    "crop_encoder.pkl"
)

joblib.dump(best_model, model_path)

joblib.dump(
    encoder,
    encoder_path
)


# ==========================================
# FINAL OUTPUT
# ==========================================

print("\n===================================")
print("Crop Recommendation Training Done!")
print("===================================")

print(
    f"Best Model Accuracy: "
    f"{best_accuracy:.2f}%"
)

print("\nModel Saved:")
print(model_path)

print("\nEncoder Saved:")
print(encoder_path)

print("\nNumber of Classes:")
print(len(encoder.classes_))

print("\nClasses:")
print(list(encoder.classes_))

print("===================================")