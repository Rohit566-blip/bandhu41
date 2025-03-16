import os
import nltk
import json
import pickle
import numpy as np
import random
from nltk.stem import WordNetLemmatizer
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Set base directory
basedir = os.path.abspath(os.path.dirname(__file__))
chatbot_dir = os.path.join(basedir, 'chatbot')

# Ensure chatbot directory exists
os.makedirs(chatbot_dir, exist_ok=True)

# Load intents file (Check if exists)
data_file = os.path.join(chatbot_dir, 'data.json')
if not os.path.exists(data_file):
    raise FileNotFoundError(f"Error: {data_file} not found!")

with open(data_file, 'r') as f:
    intents = json.load(f)

# Initialize data containers
words = []
classes = []
documents = []
ignore_words = ['?', '!', '.', ',', "'s", "'m"]

# Process intents file
for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)  # Tokenize sentence
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize, lowercase, and remove duplicates
words = sorted(set([lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words and w.isalpha()]))
classes = sorted(set(classes))

# Save processed words and classes
pickle.dump(words, open(os.path.join(chatbot_dir, 'texts.pkl'), 'wb'))
pickle.dump(classes, open(os.path.join(chatbot_dir, 'labels.pkl'), 'wb'))

# Prepare training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = [0] * len(words)
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in doc[0] if word.isalpha()]
    for word in pattern_words:
        if word in words:
            bag[words.index(word)] = 1
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# Shuffle and convert to NumPy arrays
random.shuffle(training)
training = np.array(training, dtype=object)

train_x = np.array(list(training[:, 0]), dtype=float)
train_y = np.array(list(training[:, 1]), dtype=float)

# Define neural network model
model = Sequential([
    Dense(256, input_shape=(len(train_x[0]),), activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(train_y[0]), activation='softmax')
])

# Compile model with Adam optimizer
model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True), metrics=['accuracy'])

# Train model with early stopping
early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=8, verbose=1, callbacks=[early_stopping])

# Save the trained model
model.save(os.path.join(chatbot_dir, 'model.h5_
