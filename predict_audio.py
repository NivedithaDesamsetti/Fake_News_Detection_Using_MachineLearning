import librosa
import numpy as np
import joblib
import pygame
import time

# Load trained model
model = joblib.load("audio_detection_model.pkl")

# Function to extract audio features
def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=16000, duration=3.0)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

# Ask user for audio file
file_path = input("Enter audio file path: ")

# 🎧 Play audio automatically
print("\nPlaying audio...")

pygame.mixer.init()
pygame.mixer.music.load(file_path)
pygame.mixer.music.play()

# Wait until audio finishes
while pygame.mixer.music.get_busy():
    time.sleep(1)

# Extract features
features = extract_features(file_path)
features = np.array(features).reshape(1, -1)

# Predict
prediction = model.predict(features)

print("\nPrediction Result:")

if prediction[0] == 1:
    print("✅ REAL audio")
else:
    print("⚠ FAKE audio")