import os
import nltk
import pickle
import numpy as np
import requests
from keras.models import load_model
import json
import random
from flask import Flask, render_template, request, jsonify
import mysql.connector
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup

# Download necessary NLTK data
nltk.download('popular')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Google Drive model link
MODEL_PATH = os.path.join(os.getcwd(), 'chatbot', 'model.h5')
MODEL_URL = "https://drive.google.com/uc?id=1QMySi59lofY2zJFOD4PqCDipw0Le9EH9"

# Ensure chatbot directory exists
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Download the model if it doesn't exist
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    response = requests.get(MODEL_URL, stream=True)
    with open(MODEL_PATH, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    print("Model download complete.")

# Load the model
model = load_model(MODEL_PATH)
print("Model loaded successfully.")

# Load chatbot data
intents = json.loads(open(os.path.join(os.getcwd(), 'chatbot', 'data.json')).read())
words = pickle.load(open(os.path.join(os.getcwd(), 'chatbot', 'texts.pkl'), 'rb'))
classes = pickle.load(open(os.path.join(os.getcwd(), 'chatbot', 'labels.pkl'), 'rb'))

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
