import streamlit as st
import sqlite3
import joblib
import numpy as np
import cv2
import librosa
import string
import nltk
from skimage.feature import hog
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -------------------------------
# DOWNLOAD NLTK (only first time)
# -------------------------------
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# -------------------------------
# TEXT PREPROCESSING (IMPORTANT)
# -------------------------------
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="🧠",
    layout="wide"
)

# -------------------------------
# DATABASE
# -------------------------------
conn = sqlite3.connect("users.db", check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS users(
username TEXT,
password TEXT
)
""")
conn.commit()

# -------------------------------
# LOGIN FUNCTIONS
# -------------------------------
def register(username,password):
    c.execute("INSERT INTO users VALUES (?,?)",(username,password))
    conn.commit()

def login(username,password):
    data = c.execute(
        "SELECT * FROM users WHERE username=? AND password=?",
        (username,password)
    ).fetchall()
    return len(data) > 0

# -------------------------------
# LOAD MODELS
# -------------------------------
text_model = joblib.load("logistic_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

image_model = joblib.load("image_model.pkl")
scaler = joblib.load("scaler.pkl")

audio_model = joblib.load("audio_detection_model.pkl")

# -------------------------------
# AUDIO FEATURE FUNCTION
# -------------------------------
def extract_audio_features(file):
    audio, sr = librosa.load(file, sr=16000, duration=3)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    features = np.mean(mfccs.T, axis=0)
    return features.reshape(1,-1)

# -------------------------------
# SESSION
# -------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# -------------------------------
# LOGIN PAGE
# -------------------------------
if st.session_state.logged_in == False:

    st.title("Fake News Detection Using Machine Learning")

    menu = st.sidebar.selectbox("Menu",["Login","Register"])

    if menu == "Register":

        st.subheader("Create Account")

        new_user = st.text_input("Username")
        new_pass = st.text_input("Password",type="password")

        if st.button("Register"):
            register(new_user,new_pass)
            st.success("Account created successfully")

    if menu == "Login":

        st.subheader("Login")

        username = st.text_input("Username")
        password = st.text_input("Password",type="password")

        if st.button("Login"):

            if login(username,password):
                st.session_state.logged_in = True
                st.success("Login Successful")
                st.rerun()

            else:
                st.error("Invalid Login")

# -------------------------------
# MAIN APP
# -------------------------------
else:

    st.title("Fake News Detection Using Machine Learning")

    menu = st.sidebar.selectbox(
        "Select Detection Type",
        ["Text Detection","Image Detection","Audio Detection"]
    )

    # -------------------------------
    # TEXT DETECTION (FIXED VERSION)
    # -------------------------------
    if menu == "Text Detection":

        st.header("Fake News Detection (Text)")

        text = st.text_area("Enter News Text")

        if st.button("Predict"):

            clean_text = preprocess(text)

            vect = vectorizer.transform([clean_text])

            prediction = text_model.predict(vect)[0]

            if prediction == 1:
                st.success("Real News")
            else:
                st.error("Fake News")

    # -------------------------------
    # IMAGE DETECTION
    # -------------------------------
    if menu == "Image Detection":

        st.header("Fake Image Detection")

        file = st.file_uploader("Upload Image",type=["jpg","png","jpeg"])

        if file is not None:

            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes,1)

            st.image(img,channels="BGR")

            img = cv2.resize(img,(128,128))
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            features = hog(
                gray,
                orientations=12,
                pixels_per_cell=(8,8),
                cells_per_block=(3,3),
                block_norm='L2-Hys'
            )

            features = np.array(features).reshape(1,-1)
            features = scaler.transform(features)

            prediction = image_model.predict(features)[0]

            if prediction == 1:
                st.success("Real Image")
            else:
                st.error("Fake Image")

    # -------------------------------
    # AUDIO DETECTION
    # -------------------------------
    if menu == "Audio Detection":

        st.header("Fake Audio Detection")

        audio_file = st.file_uploader("Upload Audio",type=["wav","mp3"])

        if audio_file is not None:

            features = extract_audio_features(audio_file)

            prediction = audio_model.predict(features)[0]

            if prediction == 1:
                st.success("Real Audio")
            else:
                st.error("Fake Audio")

    # -------------------------------
    # LOGOUT
    # -------------------------------
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()