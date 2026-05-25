import cv2
import joblib
import numpy as np
from skimage.feature import hog

# Load model and scaler
model = joblib.load("image_model.pkl")
scaler = joblib.load("scaler.pkl")

# Input image
image_path = input("Enter image path: ")

# Read image
img = cv2.imread(image_path)

if img is None:
    print("Image not found!")
    exit()

# Resize image (same as training)
img = cv2.resize(img, (128, 128))

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Extract HOG features (same parameters as training)
features = hog(
    gray,
    orientations=12,
    pixels_per_cell=(8,8),
    cells_per_block=(3,3),
    block_norm='L2-Hys'
)

# Convert to numpy
features = np.array(features)

# Reshape for model
features = features.reshape(1, -1)

# Scale
features = scaler.transform(features)

# Predict
prediction = model.predict(features)

if prediction[0] == 1:
    print("✅ This image is REAL")
else:
    print("❌ This image is FAKE")