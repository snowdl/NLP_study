#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import numpy as np

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Activation, Dense, Permute, Dropout
from tensorflow.keras.layers import add, dot, concatenate, LSTM, Embedding

import matplotlib.pyplot as plt


# --- Set paths safely relative to this script ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '12_data'))

train_data_path = os.path.join(DATA_DIR, 'train_qa.txt')
test_data_path = os.path.join(DATA_DIR, 'test_qa.txt')


# --- Load training and testing data ---
with open(train_data_path, "rb") as fp:
    train_data = pickle.load(fp)

with open(test_data_path, "rb") as fp:
    test_data = pickle.load(fp)


# --- Combine all data for vocabulary building ---
all_data = test_data + train_data


# --- Build vocabulary set ---
vocab = set()
for story, question, _ in all_data:
    vocab = vocab.union(set(story))
    vocab = vocab.union(set(question))

# Add special tokens 'no' and 'yes' to vocabulary
vocab.add('no')
vocab.add('yes')

vocab_size = len(vocab) + 1  # Add 1 for padding token (0)


# --- Calculate maximum lengths of stories and questions ---
max_story_len = max(len(data[0]) for data in all_data)
max_question_len = max(len(data[1]) for data in all_data)


# --- Initialize and fit tokenizer on the vocabulary ---
tokenizer = Tokenizer(filters=[])
tokenizer.fit_on_texts(vocab)


# --- Separate train and test data into components ---
train_story_text, train_question_text, train_answers = zip(*train_data)
test_story_text, test_question_text, test_answers = zip(*test_data)


# --- Define function to vectorize story, question, and answers ---
def vectorize_stories(data, word_index=tokenizer.word_index, 
                      max_story_len=max_story_len, max_question_len=max_question_len):
    X, Xq, Y = [], [], []

    for story, query, answer in data:
        # If input is a string, split it into words
        if isinstance(story, str):
            story = story.split()
        if isinstance(query, str):
            query = query.split()

        # Convert words to their integer indices in the vocabulary
        x = [word_index[word.lower()] for word in story if word.lower() in word_index]
        xq = [word_index[word.lower()] for word in query if word.lower() in word_index]

        # Create one-hot encoding for the answer
        y = np.zeros(len(word_index) + 1)
        y[word_index[answer]] = 1

        X.append(x)
        Xq.append(xq)
        Y.append(y)

    # Pad sequences to max length for stories and questions and return arrays
    return (pad_sequences(X, maxlen=max_story_len),
            pad_sequences(Xq, maxlen=max_question_len),
            np.array(Y))


# --- Vectorize training and testing data ---
inputs_train, queries_train, answers_train = vectorize_stories(train_data)
inputs_test, queries_test, answers_test = vectorize_stories(test_data)


# --- Build the model ---
input_sequence = Input((max_story_len,))
question = Input((max_question_len,))

# Input encoder m: embeds the story sequence into 64-dimensional vectors + dropout
input_encoder_m = Sequential([
    Embedding(input_dim=vocab_size, output_dim=64),
    Dropout(0.3)
])

# Input encoder c: embeds the story sequence into vectors of size max_question_len + dropout
input_encoder_c = Sequential([
    Embedding(input_dim=vocab_size, output_dim=max_question_len),
    Dropout(0.3)
])

# Question encoder: embeds the question sequence into 64-dimensional vectors + dropout
question_encoder = Sequential([
    Embedding(input_dim=vocab_size, output_dim=64),
    Dropout(0.3)
])

# Encode inputs
input_encoded_m = input_encoder_m(input_sequence)  # Shape: (samples, story_max_len, 64)
input_encoded_c = input_encoder_c(input_sequence)  # Shape: (samples, story_max_len, question_max_len)
question_encoded = question_encoder(question)      # Shape: (samples, question_max_len, 64)

# Compute dot product (attention mechanism) between story and question embeddings
match = dot([input_encoded_m, question_encoded], axes=(2, 2))  # Shape: (samples, story_max_len, question_max_len)
match = Activation('softmax')(match)  # Apply softmax to get attention weights

# Add attention weights to input_encoder_c output and permute dimensions
response = add([match, input_encoded_c])                      # Shape: (samples, story_max_len, question_max_len)
response = Permute((2, 1))(response)                          # Shape: (samples, question_max_len, story_max_len)

# Concatenate response and question encoding along the last axis
answer = concatenate([response, question_encoded])            # Shape: (samples, question_max_len, story_max_len + 64)

# Use LSTM to reduce sequence to a vector of size 32
answer = LSTM(32)(answer)

# Apply dropout for regularization
answer = Dropout(0.5)(answer)

# Output dense layer with vocab size, to predict answer word distribution
answer = Dense(vocab_size)(answer)

# Apply softmax to output probabilities
answer = Activation('softmax')(answer)

# Define and compile the model with rmsprop optimizer and categorical crossentropy loss
model = Model([input_sequence, question], answer)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()


# --- Train the model ---
history = model.fit(
    [inputs_train, queries_train],
    answers_train,
    batch_size=32,
    epochs=120,
    validation_data=([inputs_test, queries_test], answers_test)
)

# --- Save the trained model ---
model.save('chatbot_120_epochs.h5')


# --- Plot training and validation accuracy ---
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()


# --- Load model weights and evaluate test predictions ---
model.load_weights('chatbot_120_epochs.h5')
pred_results = model.predict([inputs_test, queries_test])

# Get the index of the predicted answer with highest probability for the first test example
val_max = np.argmax(pred_results[0])

# Find the corresponding word in the tokenizer vocabulary
for key, val in tokenizer.word_index.items():
    if val == val_max:
        predicted_word = key
        break

print("Predicted answer:", predicted_word)
print("True answer:", test_data[0][2])


# --- Example: Predict answer for a custom story and question ---
my_story = "John left the kitchen . Sandra dropped the football in the garden ."
my_question = "Is the football in the garden ?"
mydata = [(my_story, my_question, 'yes')]

my_story_vec, my_question_vec, my_answer_vec = vectorize_stories(mydata)

pred_results = model.predict([my_story_vec, my_question_vec])
val_max = np.argmax(pred_results[0])

for key, val in tokenizer.word_index.items():
    if val == val_max:
        predicted_answer = key
        break

print("User input predicted answer:", predicted_answer)
