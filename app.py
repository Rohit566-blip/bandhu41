import os
import nltk
import pickle
import numpy as np
import requests
import json
import random
from keras.models import load_model
from flask import Flask, render_template, request, jsonify
import mysql.connector
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup

# Download necessary NLTK data
nltk.download('popular')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Google Drive file links
MODEL_URL = "https://drive.google.com/uc?id=1QMySi59lofY2zJFOD4PqCDipw0Le9EH9"
DATA_URL = "https://drive.google.com/uc?id=1ChW3P16BGe2PCjt6HOBNdEZ-HlAcfD4j"
LABELS_URL = "https://drive.google.com/uc?id=1YZIFB--oQVvUsJZOQUtbrhHdq_Lu1xWA"
TEXTS_URL = "https://drive.google.com/uc?id=1BdDCmrdzS9scIBmbS7YKyfQg_oZq29gD"

# Define file paths
MODEL_PATH = os.path.join(os.getcwd(), 'chatbot', 'model.h5')
DATA_PATH = os.path.join(os.getcwd(), 'chatbot', 'data.json')
LABELS_PATH = os.path.join(os.getcwd(), 'chatbot', 'labels.pkl')
TEXTS_PATH = os.path.join(os.getcwd(), 'chatbot', 'texts.pkl')

# Ensure chatbot directory exists
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

def download_file(url, path, file_desc):
    """Download a file from Google Drive and save it locally."""
    try:
        print(f"Downloading {file_desc}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        
        print(f"{file_desc} download complete.")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {file_desc}: {e}")

# Download required files if missing
if not os.path.exists(MODEL_PATH):
    download_file(MODEL_URL, MODEL_PATH, "model.h5")

if not os.path.exists(DATA_PATH):
    download_file(DATA_URL, DATA_PATH, "data.json")

if not os.path.exists(LABELS_PATH):
    download_file(LABELS_URL, LABELS_PATH, "labels.pkl")

if not os.path.exists(TEXTS_PATH):
    download_file(TEXTS_URL, TEXTS_PATH, "texts.pkl")

# Load the model
model = load_model(MODEL_PATH)
print("MODEL loaded successfully.")

# Load chatbot data
with open(DATA_PATH, 'r', encoding='utf-8') as file:
    intents = json.load(file)
words = pickle.load(open(TEXTS_PATH, 'rb'))
classes = pickle.load(open(LABELS_PATH, 'rb'))

# MySQL database connection configuration
db_config = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', 'Rohit41'),
    'database': os.getenv('DB_NAME', 'chatbot')
}

def connect_db():
    try:
        conn = mysql.connector.connect(**db_config)
        print("Database connection established.")
        return conn
    except mysql.connector.Error as err:
        print(f"Database connection error: {err}")
        return None

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [1 if w in sentence_words else 0 for w in words]
    return np.array(bag)

def predict_class(sentence):
    try:
        p = bow(sentence, words)
        res = model.predict(np.array([p]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return classes[results[0][0]] if results else None
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None

def getResponse(intent):
    if not intent:
        return "I'm sorry, I didn't understand that."
    for i in intents['intents']:
        if i['tag'] == intent:
            return random.choice(i['responses'])
    return "I'm sorry, I didn't understand that."

def chatbot_response(msg):
    intent = predict_class(msg)
    return getResponse(intent)

# Flask app setup
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return jsonify(chatbot_response(userText))

@app.route('/check_db_connection', methods=['GET'])
def check_db_connection():
    conn = connect_db()
    if conn:
        conn.close()
        return jsonify({"status": "success", "message": "Database connection is healthy."}), 200
    else:
        return jsonify({"status": "error", "message": "Failed to connect to the database."}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)
