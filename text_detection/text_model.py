import os
import pickle

# Load model and vectorizer
MODEL_PATH = os.path.join("models", "text_model.pkl")
VECTORIZER_PATH = os.path.join("models", "vectorizer.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)


def predict_text(news_text):
    """
    Predict whether given news text is Real or Fake
    """

    if not news_text.strip():
        return "Please enter valid text"

    # Transform text using saved vectorizer
    transformed_text = vectorizer.transform([news_text])

    # Predict
    prediction = model.predict(transformed_text)

    if prediction[0] == 1:
        return "Fake News ❌"
    else:
        return "Real News ✅"