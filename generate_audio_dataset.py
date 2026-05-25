import os
import pandas as pd
from gtts import gTTS

MAX_SAMPLES = 50   # keep small for speed
MAX_TEXT_LENGTH = 250

os.makedirs("audio_dataset/real", exist_ok=True)
os.makedirs("audio_dataset/fake", exist_ok=True)

print("Loading CSV files...")

true_df = pd.read_csv("True.csv")
fake_df = pd.read_csv("Fake.csv")

text_column = "text" if "text" in true_df.columns else true_df.columns[0]
print("Using text column:", text_column)

true_df = true_df.head(MAX_SAMPLES)
fake_df = fake_df.head(MAX_SAMPLES)

def generate_audio(df, label):
    for i, row in df.iterrows():
        text = str(row[text_column])[:MAX_TEXT_LENGTH]
        file_path = f"audio_dataset/{label}/{label}_{i}.mp3"

        if not os.path.exists(file_path):
            tts = gTTS(text=text, lang='en')
            tts.save(file_path)

        print(f"{label} audio {i} generated")

print("Generating REAL audio...")
generate_audio(true_df, "real")

print("Generating FAKE audio...")
generate_audio(fake_df, "fake")

print("✅ Audio dataset ready!")