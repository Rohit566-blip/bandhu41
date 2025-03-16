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

# Set base directory
basedir = os.path.abspath(os.path.dirname(__file__))

# Define paths
model_path = os.path.join(basedir, 'chatbot', 'model.h5')
data_path = os.path.join(basedir, 'chatbot', 'data.json')
texts_path = os.path.join(basedir, 'chatbot', 'texts.pkl')
labels_path = os.path.join(basedir, 'chatbot', 'labels.pkl')

# Load the model and data
try:
    model = load_model(model_path)
    intents = json.loads(open(data_path).read())
    words = pickle.load(open(texts_path, 'rb'))
    classes = pickle.load(open(labels_path, 'rb'))
    print("✅ Model and data loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model or data: {e}")
    model = None
    intents = None
    words = []
    classes = []

# MySQL database connection configuration
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
        print("✅ Database connection established.")
        return conn
    except mysql.connector.Error as err:
        print(f"❌ Database connection error: {err}")
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
                if show_details:
                    print(f"found in bag: {w}")
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
        print(f"❌ Error in prediction: {e}")
        return None
    
def get_sentiment_response(sentiment):
    """Lookup the appropriate response based on sentiment"""
    for i in intents['intents']:
        if i['tag'] == 'sentiment':
            if sentiment == 'positive':
                return random.choice(i['responses'][0]['positive'])
            elif sentiment == 'negative':
                return random.choice(i['responses'][1]['negative'])
            elif sentiment == 'neutral':
                return random.choice(i['responses'][2]['neutral'])    
    
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
    Handles general intents and specific cases like sentiment analysis and timetable queries.
    """
    # Predict intent
    intent = predict_class(msg, model)
    
    # Check if the intent is about the timetable
    if intent == "timetable":
        year = extract_year(msg)
        day = extract_day(msg)

        # Check if both year and day are specified
        if year and day:
            return get_timetable(year, day)

        # If only year is found
        if year:
            return f"You specified year {year}. Please tell me which day."

        # If only day is found
        if day:
            return f"You specified day {day}. Please tell me which year."

        # If neither year nor day is found, ask for both
        return "For which year (second, third, or final) and day would you like to see the timetable?"

    # Handle general responses
    res = getResponse(intent, intents)

    # If a response is found, perform sentiment analysis
    if res:
        sentiment_response = requests.post("https://sentiment-analysis-vsj7.onrender.com", data={"text": msg})
        soup = BeautifulSoup(sentiment_response.text, 'html.parser')
       
        # Extract the sentiment text using a more specific selector
        sentiment = soup.find('div')
        if sentiment:
            sentiment = sentiment.text.strip()
        
        # Generate a response based on sentiment (custom logic here)
        sentiment_response = get_sentiment_response(sentiment)
        return f"{res} {sentiment_response}"
    
    # If no response is found, return a default fallback response
    return "I'm sorry, I didn't understand that."

def extract_year(message):
    """Extract the year (second, third, final) from the message"""
    if "second" in message.lower():
        return "second"
    elif "third" in message.lower():
        return "third"
    return None

def extract_day(message):
    """Extract the day (Monday, Tuesday, etc.) from the message"""
    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    for day in days:
        if day in message.lower():
            return day.capitalize()  # Return the day with the first letter capitalized
    return None

def fetch_timetable_from_db(year, day):
    """Fetch the timetable for the given year and day from the MySQL database"""
    conn = connect_db()
    if conn is None:
        return None
    with conn.cursor(dictionary=True) as cursor:
        if year == 'second':
            query = "SELECT * FROM timetable_second_year WHERE day = %s"
        elif year == 'third':
            query = "SELECT * FROM timetable_third_year WHERE day = %s"
        
        cursor.execute(query, (day,))
        results = cursor.fetchall()
    conn.close()
    return results if results else None

def get_timetable(year, day):
    """Retrieve timetable for a specific year and day"""
    timetable_data = fetch_timetable_from_db(year, day)
    if timetable_data:
        timetable_str = f"Timetable for {year} year on {day}:\n"
        for entry in timetable_data:
            timetable_str += f"{entry['time']}: {entry['subject']}\n"
        return timetable_str
    else:
        return f"No timetable found for {year} year on {day}."

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

@app.route('/timetable', methods=['GET'])
def timetable():
    year = request.args.get('year')
    day = request.args.get('day')
    if not year or not day:
        return jsonify({'error': 'Missing parameters'}), 400

    timetable_data = fetch_timetable_from_db(year, day)
    if timetable_data:
        return jsonify({'timetable': timetable_data}), 200
    else:
        return jsonify({'error': 'No timetable found'}), 404
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
