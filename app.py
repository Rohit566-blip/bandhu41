import os
import nltk
import pickle
import numpy as np
from keras.models import load_model
import json
import random
from flask import Flask, render_template, request, jsonify
import mysql.connector
from mysql.connector import pooling
from nltk.stem import WordNetLemmatizer
import gdown  # For downloading missing files

# Download necessary NLTK data
nltk.download('popular')

# Initialize Flask app
app = Flask(__name__)

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Set base directory for file paths
basedir = os.path.abspath(os.path.dirname(__file__))

# Define chatbot paths
model_path = os.path.join(basedir, 'chatbot', 'model.h5')
data_path = os.path.join(basedir, 'chatbot', 'data.json')
texts_path = os.path.join(basedir, 'chatbot', 'texts.pkl')
labels_path = os.path.join(basedir, 'chatbot', 'labels.pkl')

# Google Drive links (Replace with your actual file ID)
MODEL_URL = "https://drive.google.com/uc?id=YOUR_MODEL_FILE_ID"

# Ensure chatbot directory exists
os.makedirs(os.path.dirname(model_path), exist_ok=True)

# Download model if not found
if not os.path.exists(model_path):
    print("🔴 Model not found! Downloading from Google Drive...")
    gdown.download(MODEL_URL, model_path, quiet=False)

# Load the chatbot model
try:
    model = load_model(model_path)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Model loading error: {e}")
    model = None

# Load intents data
try:
    with open(data_path) as f:
        intents = json.load(f)
except FileNotFoundError:
    print("⚠️ 'data.json' file not found!")
    intents = None

# Load tokenizer and classes
try:
    words = pickle.load(open(texts_path, 'rb'))
    classes = pickle.load(open(labels_path, 'rb'))
except FileNotFoundError as e:
    print(f"⚠️ Missing tokenizer files: {e}")
    words, classes = [], []

# Database configuration (Use environment variables for security)
db_config = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', 'rohit41'),
    'database': os.getenv('DB_NAME', 'chatbot'),
    'pool_name': 'mypool',
    'pool_size': 5
}

# Connection pool for MySQL
try:
    db_pool = pooling.MySQLConnectionPool(**db_config)
    print("✅ Database connection pool created successfully.")
except Exception as e:
    print(f"❌ Database pool error: {e}")
    db_pool = None

# Function to get DB connection
def get_db_connection():
    try:
        return db_pool.get_connection() if db_pool else None
    except mysql.connector.Error as err:
        print(f"❌ Error getting DB connection: {err}")
        return None

# Tokenization and Lemmatization
def clean_up_sentence(sentence):
    """Tokenize and lemmatize the input sentence"""
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Bag-of-words conversion
def bow(sentence, words):
    """Create a bag-of-words representation of the input sentence"""
    sentence_words = clean_up_sentence(sentence)
    bag = [1 if w in sentence_words else 0 for w in words]
    return np.array(bag)

# Predict chatbot response
def predict_class(sentence):
    """Predict the class of the input sentence using the trained model"""
    if model is None:
        return None

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

# Get response from intents.json
def get_response(ints):
    """Get a random response for the predicted intent"""
    if not ints or not intents:
        return "I'm sorry, I didn't understand that."
    for intent in intents.get('intents', []):
        if intent['tag'] == ints:
            return random.choice(intent['responses'])
    return "I'm sorry, I didn't understand that."

# Main chatbot response function
def chatbot_response(msg):
    """Generate a chatbot response for the user message"""
    intent = predict_class(msg)
    return get_response(intent)

# Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["GET"])
def get_bot_response():
    user_text = request.args.get('msg')
    return jsonify({"response": chatbot_response(user_text)})

@app.route('/check_db_connection', methods=['GET'])
def check_db_connection():
    conn = get_db_connection()
    if conn:
        conn.close()
        return jsonify({"status": "success", "message": "Database connection is healthy."}), 200
    else:
        return jsonify({"status": "error", "message": "Failed to connect to the database."}), 500

# Start the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
