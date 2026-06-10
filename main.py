import numpy as np
import tensorflow as ts
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import string
import streamlit as st


## load word index of imdb dataset

word_index = imdb.get_word_index()
reverse_word_index = {value: key for (key, value) in word_index.items()}

## load model

model = load_model('simple_rnn_imdb.h5')



## helper function

def decode_review(text):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in text])

def preprocess_text(text):
  # Clean the text: lowercase and remove punctuation
  text = text.lower()
  text = text.translate(str.maketrans('', '', string.punctuation))
  words = text.split()

  # Convert words to their integer representations, shifting by 3.
  # 0: padding, 1: start-of-sequence, 2: unknown
  # Words from word_index are 1-indexed. We add 3 to them.
  # Words not found in vocabulary map to 2 (unknown token).
  encoded_review = [word_index.get(word, 2) + 3 for word in words]

  # Add the start-of-sequence token (1) at the beginning
  encoded_review = [1] + encoded_review

  # Pad the sequence to maxlen, applying padding at the end
  return sequence.pad_sequences([encoded_review], maxlen=500, padding='post')

## prediction dunction

def predict_review(review):
  encoded_review = preprocess_text(review)
  prediction = model.predict(encoded_review)

  sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
  confidence = prediction[0][0] if sentiment == 'Positive' else 1 - prediction[0][0]

  return sentiment, confidence, prediction[0][0]

## streamlit app

st.title('IMDB Sentiment Analysis')
st.write('Enter a review')

## user input
user_input = st.text_area('Moview Review')
if st.button('Predict'):
  preprocessed_input = preprocess_text(user_input)
  ## make prediction
  sentiment, confidence, prediction = predict_review(preprocessed_input)

  st.write(f'Sentiment: {sentiment}')
  st.write(f'Confidence: {confidence:.2f}')
  st.write(f'Prediction: {prediction:.2f}')