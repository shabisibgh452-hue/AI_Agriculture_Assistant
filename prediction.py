import os
import sys
import numpy as np
import tensorflow as tf

# ==========================
# Root Path
# ==========================
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DISEASE_MODEL

# ==========================
# Load Model
# ==========================
model = tf.keras.models.load_model(DISEASE_MODEL)

# ==========================
# Class Names
# ==========================
class_names = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# ==========================
# Image Path
# ==========================
image_path = input("Enter leaf image path: ")

# ==========================
# Check Image
# ==========================
if not os.path.exists(image_path):
    print("Image not found!")
    exit()

# ==========================
# Load Image
# ==========================
img = tf.keras.utils.load_img(
    image_path,
    target_size=(128, 128)      # Same as training.py
)

img_array = tf.keras.utils.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

# ==========================
# Prediction
# ==========================
prediction = model.predict(img_array, verbose=0)

predicted_class = np.argmax(prediction)
confidence = np.max(prediction) * 100

print("\n===================================")
print("Predicted Disease :", class_names[predicted_class])
print(f"Confidence : {confidence:.2f}%")
print("===================================")