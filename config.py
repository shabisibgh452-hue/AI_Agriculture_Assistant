import os

# ==========================================
# BASE DIRECTORY
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# FOLDERS
# ==========================================
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
MODEL_DIR = os.path.join(BASE_DIR, "models")
ENCODER_DIR = os.path.join(BASE_DIR, "encoders")

# ==========================================
# CREATE FOLDERS (IF NOT EXISTS)
# ==========================================
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ENCODER_DIR, exist_ok=True)

# ==========================================
# DATASET PATHS
# ==========================================
CROP_DATASET = os.path.join(DATASET_DIR, "crop_recommendation.csv")
FERTILIZER_DATASET = os.path.join(DATASET_DIR, "Fertilizer Prediction.csv")
YIELD_DATASET = os.path.join(DATASET_DIR, "yield_df.csv")
PESTICIDES_DATASET = os.path.join(DATASET_DIR, "pesticides.csv")
RAINFALL_DATASET = os.path.join(DATASET_DIR, "rainfall.csv")
TEMP_DATASET = os.path.join(DATASET_DIR, "temp.csv")

# ==========================================
# MODEL PATHS
# ==========================================
CROP_MODEL = os.path.join(MODEL_DIR, "crop_model.pkl")
FERTILIZER_MODEL = os.path.join(MODEL_DIR, "fertilizer_model.pkl")
YIELD_MODEL = os.path.join(MODEL_DIR, "yield_model.pkl")
# ==========================================
# ENCODER & SCALER PATHS
# ==========================================
SOIL_ENCODER = os.path.join(ENCODER_DIR, "soil_encoder.pkl")
CROP_ENCODER = os.path.join(ENCODER_DIR, "crop_encoder.pkl")
FERTILIZER_ENCODER = os.path.join(ENCODER_DIR, "fertilizer_encoder.pkl")
SCALER = os.path.join(ENCODER_DIR, "scaler.pkl")
# ==========================================
# DISEASE PREDICTION DATASET
# ==========================================

DISEASE_DATASET = os.path.join(DATASET_DIR, "PlantVillage", "PlantVillage")
DISEASE_TRAIN = os.path.join(DISEASE_DATASET, "train")
DISEASE_VALID = os.path.join(DISEASE_DATASET, "valid")

# ==========================================
# DISEASE MODEL
# ==========================================

DISEASE_MODEL = os.path.join(MODEL_DIR, "disease_model.keras")