import nltk
import os
import json
import pickle
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Define the path to your project folder, this is used for relative paths
project_root = os.getcwd()

# Load the data (using relative path)
data_file_path = os.path.join(project_root, 'chatbot', 'data.json')
data_file = open(data_file_path).read()
intents = json.loads(data_file)

words = []
classes = []
documents = []
ignore_words = ['?', '!']

# Process the intents and create the word list and classes
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

# Lemmatize and lower each word, and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# Sort classes
classes = sorted(list(set(classes)))

# Print the lengths of the processed data
print(f"{len(documents)} documents")
print(f"{len(classes)} classes: {classes}")
print(f"{len(words)} unique lemmatized words: {words}")

# Save words and classes to pickle files (using relative path)
pickle.dump(words, open(os.path.join(project_root, 'chatbot', 'texts.pkl'), 'wb'))
pickle.dump(classes, open(os.path.join(project_root, 'chatbot', 'labels.pkl'), 'wb'))

# Create our training data
training = []
output_empty = [0] * len(classes)

# Create bag of words for each sentence
for doc in documents:
    # Initialize our bag of words
    bag = [0] * len(words)
    # List of tokenized words for the pattern
    pattern_words = doc[0]
    # Lemmatize each word to create a base word
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # Create the bag of words array with 1 if word match found in current pattern
    for pattern_word in pattern_words:
        for i, w in enumerate(words):
            if w == pattern_word:
                bag[i] = 1
    # Output is a '0' for each tag and '1' for the current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# Shuffle the training data and convert to np.array
random.shuffle(training)
training = np.array(training, dtype=object)
train_x = np.array(list(training[:, 0]), dtype=float)
train_y = np.array(list(training[:, 1]), dtype=float)

print("Training data created")

# Create the model - 3 layers: 128 neurons in the first layer, 64 neurons in the second layer, and output layer with neurons equal to the number of classes
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile the model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Fit the model and save it
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# Save the model (using relative path)
model_file_path = os.path.join(project_root, 'chatbot', 'model.h5')
model.save(model_file_path)

print("Model created and saved")
