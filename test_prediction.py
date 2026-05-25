import joblib
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

# Load model
model = joblib.load("logistic_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

print("Enter News Article:\n")
news = input()

news = preprocess(news)
news_tfidf = vectorizer.transform([news])

prediction = model.predict(news_tfidf)

if prediction[0] == 1:
    print("\nPrediction: REAL NEWS ✅")
else:
    print("\nPrediction: FAKE NEWS ❌")