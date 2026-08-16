import os
import sys
import tensorflow as tf

# Root path add
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DISEASE_TRAIN, DISEASE_VALID, DISEASE_MODEL

# ==========================
# Parameters
# ==========================
IMG_SIZE = (128, 128)
BATCH_SIZE = 8
EPOCHS = 3

# ==========================
# Load Training Dataset
# ==========================
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DISEASE_TRAIN,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True
)

# ==========================
# Load Validation Dataset
# ==========================
valid_dataset = tf.keras.utils.image_dataset_from_directory(
    DISEASE_VALID,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

# ==========================
# Normalize Images
# ==========================
normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

train_dataset = train_dataset.map(
    lambda x, y: (normalization_layer(x), y)
)

valid_dataset = valid_dataset.map(
    lambda x, y: (normalization_layer(x), y)
)

# ==========================
# CNN Model
# ==========================
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(128, 128, 3)),

    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(38, activation="softmax")
])

# ==========================
# Compile Model
# ==========================
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ==========================
# Train Model
# ==========================
history = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=EPOCHS
)

# ==========================
# Save Model
# ==========================
model.save(DISEASE_MODEL)

print("\n===================================")
print("Disease Model Trained Successfully!")
print("Model Saved At:", DISEASE_MODEL)
print("===================================")

# ==========================
# Final Accuracy
# ==========================
train_acc = history.history["accuracy"][-1]
val_acc = history.history["val_accuracy"][-1]

print(f"Training Accuracy : {train_acc*100:.2f}%")
print(f"Validation Accuracy : {val_acc*100:.2f}%")