# 🌱 AI Agriculture Assistant

AI Agriculture Assistant is a Machine Learning and Deep Learning based web application that helps farmers make better agricultural decisions.

This project includes four intelligent modules:

- 🌾 Crop Recommendation
- 🧪 Fertilizer Recommendation
- 📈 Crop Yield Prediction
- 🍃 Plant Disease Prediction

Built with:
- Python
- Streamlit
- Scikit-learn
- TensorFlow
- Pandas
- NumPy

---

# 🚀 Features

### 🌾 Crop Recommendation
Recommends the best crop based on:

- Nitrogen
- Phosphorus
- Potassium
- Temperature
- Humidity
- pH
- Rainfall

---

### 🧪 Fertilizer Recommendation
Suggests the most suitable fertilizer using:

- Temperature
- Humidity
- Moisture
- Soil Type
- Crop Type
- Nitrogen
- Potassium
- Phosphorous

---

### 📈 Crop Yield Prediction
Predicts crop yield using:

- Area
- Crop Name
- Year
- Rainfall
- Pesticides
- Temperature

---

### 🍃 Plant Disease Prediction
Detects plant diseases from leaf images using CNN Deep Learning model.

Supported crops:

- Apple
- Tomato
- Potato
- Corn
- Grape
- Peach
- Pepper
- Strawberry
- Orange
- Soybean

---

# 📁 Project Structure

```text
AI_Agriculture_Assistant/
│
├── app.py
├── config.py
├── requirements.txt
├── README.md
│
├── datasets/
│   ├── cleaned_crop_data.csv
│   ├── Fertilizer Prediction.csv
│   ├── yield_cleaned.csv
│   └── PlantVillage/
│       ├── train/
│       └── valid/
│
├── models/
│   ├── best_crop_model.pkl
│   ├── fertilizer_model.pkl
│   ├── crop_yield_model.pkl
│   └── disease_model.keras
│
├── encoders/
│   ├── crop_encoder.pkl
│   ├── soil_encoder.pkl
│   ├── fertilizer_encoder.pkl
│   ├── area_encoder.pkl
│   ├── item_encoder.pkl
│   └── scaler.pkl
│
├── crop_recommendation/
│   ├── preprocessing.py
│   ├── training.py
│   └── prediction.py
│
├── fertilizer_recommendation/
│   ├── preprocessing.py
│   ├── training.py
│   └── prediction.py
│
├── crop_yield_prediction/
│   ├── preprocessing.py
│   ├── training.py
│   └── prediction.py
│
└── disease_prediction/
    ├── preprocessing.py
    ├── training.py
    └── prediction.py
```

---

# ⚙️ Installation

Clone repository:

```bash
git clone https://github.com/YourUsername/AI_Agriculture_Assistant.git
```

Move to project folder:

```bash
cd AI_Agriculture_Assistant
```

Install libraries:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Application

```bash
streamlit run app.py
```

Application will open at:

```text
http://localhost:8501
```

---

# 🤖 Machine Learning Models

| Module | Model |
|----------|----------|
| Crop Recommendation | Random Forest |
| Fertilizer Recommendation | Random Forest |
| Crop Yield Prediction | Random Forest Regressor |
| Disease Prediction | CNN (TensorFlow) |

---

# 📊 Dataset Sources

- Kaggle
- PlantVillage Dataset

---

# 👩‍💻 Developed By

**Nazish Safdar**

BS Computer Science  
Women University Mardan

---

# 🌟 Future Improvements

- Weather API Integration
- Multi-language Support
- Mobile Application
- Cloud Deployment
- Real-time Disease Detection