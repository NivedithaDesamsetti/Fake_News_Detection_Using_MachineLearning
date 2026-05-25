import os
import cv2
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog

DATASET_PATH = "real_and_fake_face"

X = []
y = []


def process_images(folder, label):

    for file in os.listdir(folder):

        path = os.path.join(folder, file)
        img = cv2.imread(path)

        if img is None:
            continue

        # Resize image
        img = cv2.resize(img, (128, 128))

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Extract HOG features
        features = hog(
            gray,
            orientations=12,
            pixels_per_cell=(8, 8),
            cells_per_block=(3, 3),
            block_norm='L2-Hys'
        )

        X.append(features)
        y.append(label)


print("Processing REAL images...")
process_images(os.path.join(DATASET_PATH, "training_real"), 1)

print("Processing FAKE images...")
process_images(os.path.join(DATASET_PATH, "training_fake"), 0)

X = np.array(X)
y = np.array(y)

print("Total images:", len(X))


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Scaling features...")

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


print("Training model...")

model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale"
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)

print("Accuracy:", accuracy)


# Save model and scaler
joblib.dump(model, "image_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model saved successfully!")