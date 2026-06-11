import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH = "fruit_model.keras"

# =========================
# Load Model
# =========================
model = tf.keras.models.load_model(
    MODEL_PATH
)

# =========================
# Image Path
# =========================
image_path = input(
    "Enter image path: "
)

# =========================
# Load Image
# =========================
img = Image.open(image_path)

img = img.convert("RGB")

img = img.resize((224,224))

img_array = np.array(
    img,
    dtype=np.float32
)

img_array = np.expand_dims(
    img_array,
    axis=0
)

# =========================
# Prediction
# =========================
prediction = model.predict(
    img_array,
    verbose=0
)

score = float(
    prediction[0][0]
)

# =========================
# Result
# =========================
if score >= 0.5:

    label = "Non-Apple"

    confidence = score * 100

else:

    label = "Apple"

    confidence = (1 - score) * 100

print("\nResult")
print("-------------")
print("Prediction :", label)
print(
    f"Confidence : {confidence:.2f}%"
)