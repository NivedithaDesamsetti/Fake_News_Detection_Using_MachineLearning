import os
import librosa
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=16000, duration=3.0)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"⚠ Skipping corrupted file: {file_path}")
        return None


X = []
y = []

print("Processing FAKE audio...")
for file in os.listdir("audio_dataset/fake"):
    if not file.endswith(".mp3"):
        continue

    path = os.path.join("audio_dataset/fake", file)

    # Skip empty files
    if os.path.getsize(path) == 0:
        print(f"⚠ Empty file skipped: {file}")
        continue

    features = extract_features(path)

    if features is not None:
        X.append(features)
        y.append(0)


print("Processing REAL audio...")
for file in os.listdir("audio_dataset/real"):
    if not file.endswith(".mp3"):
        continue

    path = os.path.join("audio_dataset/real", file)

    if os.path.getsize(path) == 0:
        print(f"⚠ Empty file skipped: {file}")
        continue

    features = extract_features(path)

    if features is not None:
        X.append(features)
        y.append(1)


X = np.array(X)
y = np.array(y)

print("Total valid samples:", len(X))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training model...")

model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

joblib.dump(model, "audio_detection_model.pkl")
print("✅ Model saved successfully!")