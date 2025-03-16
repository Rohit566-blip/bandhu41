import os
import nltk
import pickle
import numpy as np
from keras.models import load_model
import json
import random
from flask import Flask, render_template, request, jsonify
import mysql.connector
from nltk.stem import WordNetLemmatizer
import requests
from bs4 import BeautifulSoup

# Download necessary NLTK data
nltk.download('popular')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Set base directory for file paths
basedir = os.path.abspath(os.path.dirname(__file__))

# Load the model and data (Using relative paths)
model = load_model(os.path.join(basedir, 'chatbot', 'model.h5'))
with open(os.path.join(basedir, 'chatbot', 'data.json')) as f:
    intents = json.load(f)

words = pickle.load(open(os.path.join(basedir, 'chatbot', 'texts.pkl'), 'rb'))
classes = pickle.load(open(os.path.join(basedir, 'chatbot', 'labels.pkl'), 'rb'))

# MySQL database connection configuration (Use environment variables for security)
db_config = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', 'rohit41'),
    'database': os.getenv('DB_NAME', 'chatbot')
}

def connect_db():
    """Establish a connection to the MySQL database"""
    try:
        conn = mysql.connector.connect(**db_config)
        print("Database connection established.")
        return conn
    except mysql.connector.Error as err:
        print(f"Database connection error: {err}")
        return None

def clean_up_sentence(sentence):
    """Tokenize and lemmatize the input sentence"""
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    """Create a bag-of-words representation of the input sentence"""
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, model):
    """Predict the class of the input sentence using the trained model"""
    try:
        p = bow(sentence, words, show_details=False)
        res = model.predict(np.array([p]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return classes[results[0][0]] if results else None
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None

def getResponse(ints, intents_json):
    """Get a random response for the predicted intent"""
    if not ints:
        return "I'm sorry, I didn't understand that."
    for intent in intents_json['intents']:
        if intent['tag'] == ints:
            return random.choice(intent['responses'])
    return "I'm sorry, I didn't understand that."

def chatbot_response(msg):
    """
    Generate a response from the chatbot for the given message.
    """
    intent = predict_class(msg, model)
    res = getResponse(intent, intents)
    return res if res else "I'm sorry, I didn't understand that."

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
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
