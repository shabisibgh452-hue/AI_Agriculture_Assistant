import os
import sys
import tensorflow as tf

# Root path add
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DISEASE_TRAIN, DISEASE_VALID

# Image size
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Training Dataset
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DISEASE_TRAIN,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True
)

# Validation Dataset
valid_dataset = tf.keras.utils.image_dataset_from_directory(
    DISEASE_VALID,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

# Save class names BEFORE map()
class_names = train_dataset.class_names

# Normalize Images
normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

train_dataset = train_dataset.map(
    lambda x, y: (normalization_layer(x), y)
)

valid_dataset = valid_dataset.map(
    lambda x, y: (normalization_layer(x), y)
)

print("=" * 50)
print("Disease Dataset Loaded Successfully!")
print("=" * 50)

print("Number of Classes:", len(class_names))

print("\nClasses:")
for i, cls in enumerate(class_names):
    print(f"{i+1}. {cls}")

print("\nPreprocessing Completed Successfully!")