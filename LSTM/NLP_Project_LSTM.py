#!/usr/bin/env python
# coding: utf-8

'''
Author: Ayushi Yadav 
        Nandan Prasad
        Sai Pavan

Date: 30th March 2023

Title : NLP_Project_LSTM.py

Description : The python code trains the dataset of LSTM model, 
            tests the data and gives the score of the model that is achieved.

'''



# Import libraries 

import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense, Dropout, SpatialDropout1D
from tensorflow.keras.layers import Embedding




# Input data

df = pd.read_csv("./data.csv")ra
df = df[['text','sentiment']]
print(df.shape)


df = df[df['sentiment'] != 'neutral']
print(df.shape)


df["sentiment"].value_counts()


sentiment_label = df.sentiment.factorize()
sentiment_label


# Tokenize, convert to word embeddings and pad to equal length

reviews = df.text.values
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(reviews)
vocab_size = len(tokenizer.word_index) + 1
encoded_docs = tokenizer.texts_to_sequences(reviews)
padded_sequence = pad_sequences(encoded_docs, maxlen=200)

# Print

print(tokenizer.word_index)
print(reviews[0])
print(encoded_docs[0])
print(padded_sequence[0])


# Defining the LSTM model architecture 

embedding_vector_length = 32
model = Sequential() 
model.add(Embedding(vocab_size, embedding_vector_length, input_length=200) )
model.add(SpatialDropout1D(0.25))
model.add(LSTM(50, dropout=0.5, recurrent_dropout=0.5))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid')) 
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])  
print(model.summary()) 


# Fitting the model to the data

history = model.fit(padded_sequence,sentiment_label[0],validation_split=0.2, epochs=5, batch_size=32)


# Plotting the accuracy curve

plt.plot(history.history['accuracy'], label='acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()
plt.savefig("Accuracy plot.jpg")


# Plotting the loss curve

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()
plt.savefig("Loss plot.jpg")


def predict_sentiment(text):
    tw = tokenizer.texts_to_sequences([text])
    tw = pad_sequences(tw,maxlen=200)
    prediction = int(model.predict(tw).round().item())
    print("Predicted label: ", sentiment_label[1][prediction])


test_sentence1 = "[custom data 1]"
predict_sentiment(test_sentence1)

test_sentence2 = "[custom data 2]"
predict_sentiment(test_sentence2)


