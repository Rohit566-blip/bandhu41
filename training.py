import os
import nltk
import json
import pickle
import requests
import numpy as np
import random
from nltk.stem import WordNetLemmatizer
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Google Drive file links
DATA_URL = "https://drive.google.com/uc?id=1ChW3P16BGe2PCjt6HOBNdEZ-HlAcfD4j"
LABELS_URL = "https://drive.google.com/uc?id=1YZIFB--oQVvUsJZOQUtbrhHdq_Lu1xWA"
TEXTS_URL = "https://drive.google.com/uc?id=1BdDCmrdzS9scIBmbS7YKyfQg_oZq29gD"

# Define file paths
DATA_PATH = os.path.join(os.getcwd(), 'chatbot', 'data.json')
LABELS_PATH = os.path.join(os.getcwd(), 'chatbot', 'labels.pkl')
TEXTS_PATH = os.path.join(os.getcwd(), 'chatbot', 'texts.pkl')

# Ensure chatbot directory exists
os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

def download_file(url, path, file_desc, is_json=False):
    """Download a file from Google Drive and save it locally."""
    try:
        print(f"Downloading {file_desc}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        if is_json:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(response.json(), f, indent=4)
        else:
            with open(path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
        
        print(f"{file_desc} download complete.")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {file_desc}: {e}")

# Download required files if missing
if not os.path.exists(DATA_PATH):
    download_file(DATA_URL, DATA_PATH, "data.json", is_json=True)

if not os.path.exists(LABELS_PATH):
    download_file(LABELS_URL, LABELS_PATH, "labels.pkl")

if not os.path.exists(TEXTS_PATH):
    download_file(TEXTS_URL, TEXTS_PATH, "texts.pkl")

# Load data.json
with open(DATA_PATH, 'r', encoding='utf-8') as file:
    intents = json.load(file)

# Preprocessing data
words = []
classes = []
documents = []
ignore_words = ['?', '!']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize words
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

print(f"{len(documents)} documents")
print(f"{len(classes)} classes: {classes}")
print(f"{len(words)} unique lemmatized words")

# Save words and classes
pickle.dump(words, open(TEXTS_PATH, 'wb'))
pickle.dump(classes, open(LABELS_PATH, 'wb'))

# Creating training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = [0] * len(words)
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in doc[0]]
    
    for pattern_word in pattern_words:
        for i, w in enumerate(words):
            if w == pattern_word:
                bag[i] = 1
    
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)
train_x = np.array(list(training[:, 0]), dtype=float)
train_y = np.array(list(training[:, 1]), dtype=float)

print("Training data created.")

# Build model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train and save model
hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)
model.save(os.path.join(os.getcwd(), 'chatbot', 'model.h5'), hist)

print("Model training complete and saved as 'model.h5'.")
