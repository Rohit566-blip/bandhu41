import os
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Set base directory
basedir = os.path.abspath(os.path.dirname(__file__))

# Define paths
data_path = os.path.join(basedir, 'chatbot', 'data.json')
texts_path = os.path.join(basedir, 'chatbot', 'texts.pkl')
labels_path = os.path.join(basedir, 'chatbot', 'labels.pkl')
model_path = os.path.join(basedir, 'chatbot', 'model.h5')

# Load intents data
try:
    with open(data_path) as f:
        intents = json.load(f)
    print("✅ Intents data loaded successfully!")
except Exception as e:
    print(f"❌ Error loading intents data: {e}")
    intents = None

# Initialize lists
words = []
classes = []
documents = []
ignore_words = ['?', '!']

# Process intents data
if intents:
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            # Tokenize each word
            w = nltk.word_tokenize(pattern)
            words.extend(w)
            # Add documents in the corpus
            documents.append((w, intent['tag']))
            # Add to our classes list
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    # Lemmatize and lower each word and remove duplicates
    words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
    words = sorted(list(set(words)))
    # Sort classes
    classes = sorted(list(set(classes)))

    print(len(documents), "documents")
    print(len(classes), "classes", classes)
    print(len(words), "unique lemmatized words", words)

    # Save words and classes
    pickle.dump(words, open(texts_path, 'wb'))
    pickle.dump(classes, open(labels_path, 'wb'))

    # Create our training data
    training = []
    output_empty = [0] * len(classes)

    for doc in documents:
        bag = [0] * len(words)
        pattern_words = doc[0]
        pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
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

    print("✅ Training data created")

    # Create model - 3 layers
    model = Sequential()
    model.add(Dense(128, input_shape=(len(train_x[0]), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation='softmax'))

    # Compile model
    sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # Fit the model
    hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

    # Save the model
    model.save(model_path)
    print(f"✅ Model saved to {model_path}")
else:
    print("❌ No intents data found. Exiting...")
