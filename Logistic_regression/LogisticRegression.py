'''
Author: Ayushi Yadav 
        Nandan Prasad
        Sai Pavan

Title : Final_Project.py

Description : The python module executes the logistic regression model.

Functions: It has 5 functions:
            1. clean_train_tweets(df)
            2. clean_data()
            3. vectorize()
            4. test_review()
            5. test_data()
            

'''



import numpy as np
import pandas as pd
import seaborn as sb
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import preprocessor as p
import random
import re
from stop_words import get_stop_words
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet 
import nltk
import nltk.corpus
from nltk.tokenize import word_tokenize
from nltk.lm.preprocessing import pad_both_ends
from nltk.util import bigrams
from nltk.util import everygrams
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet 

global tempArr
tempArr = []

replace_no_space = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\|)|(\()|(\))|(\[)|(\])|(/%)|(\$)|(\>)|(\<)|(\{)|(\})")
replace_with_space = re.compile("(<br\s/><br\s/?)|(-)|(/)|(:).")

df_test = pd.read_csv("Test.csv")

df_train = pd.read_csv("clean_train.csv")

x_train = df_train.text.values
y_train = df_train.label.values

x_test = df_test.text.values
y_test = df_test.label.values


def clean_train_tweets(df):

    for i in range(len(df)):
        tmpl = p.clean(df[i])                                 # preprocessor library to clean data
        tmpl = replace_no_space.sub("", tmpl.lower())        # lower casing
        tmpl = replace_with_space.sub(" ", tmpl)             # replacing with a space in case of special char
        line = tmpl.lower()
        stop_words = get_stop_words('english')               # removing stopwords
        word_tokens = list(line.split())
        # filtered_sentence = []                               # initialize empty list
        # filtered_sentence.append('</s>')
        filtered_sentence =""
        for w in word_tokens:
            if w not in stop_words:
                ps = nltk.LancasterStemmer()
                # lm = nltk.WordNetLemmatizer()
                wor = ps.stem(w)
                filtered_sentence = filtered_sentence + wor + " "
        global tempArr
        tempArr.append(filtered_sentence)                   # appending it to gloabl array
    return tempArr



def clean_data():

    df = pd.read_csv("Train.csv")
    print(df.head())
    train_reviews = clean_train_tweets(df["text"])
    # creating file
    headerList = ['text', 'label']
    filename = "clean_train.csv"
    with open(filename, 'w') as file:
        dw = csv.DictWriter(file, delimiter=',', 
                        fieldnames=headerList)
        dw.writeheader()
    for i in range(len(tempArr)):
        x = random.randint(0,1000000)
        data_to_append = ({ 'text' : [tempArr[i]], 'score' : [df.label[i]]})
        df2 = pd.DataFrame(data_to_append)
 
        # append data frame to CSV file
        df2.to_csv(filename, mode='a', index=False, header=False)



def vectorize():

# vectorize
    vectorizer = CountVectorizer()
    vectorizer.fit(x_train)
    X_train = vectorizer.transform(x_train)
    X_test = vectorizer.transform(x_test)

    classifier = LogisticRegression(max_iter = 1000)
    classifier.fit(X_train, y_train)

    score = classifier.score(X_test, y_test)
    print ("Accuracy is: ", score)

    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels= df_train.label.unique())
    df_cm = pd.DataFrame(cm, index = df_train.label.unique(), columns = df_train.label.unique())

def test_review():
    tweet = "I am good"
    vectTweet = vectorizer.transform(np.array([tweet]))
    prediction = classifier.predict(vectTweet)


def test_data():
#testing the data
    tempArr.clear()
    clean_test_df = clean_train_tweets(df_test['text'])
    Test_score = []
    for i in clean_test_df:
        sc = vectorizer.transform(np.array([i]))
        prediction = classifier.predict(sc)
        Test_score.append(prediction[0])


    headerList = ['text', 'label']
    filename = "test_result.csv"
    with open(filename, 'w') as file:
            dw = csv.DictWriter(file, delimiter=',', 
                            fieldnames=headerList)
            dw.writeheader()
    for i in range(len(Test_score)):
        x = random.randint(0,1000000)
        data_to_append = ({ 'text' : [df_test["text"][i]], 'score' : [Test_score[i]]})
        df2 = pd.DataFrame(data_to_append)
 
        # append data frame to CSV file
        df2.to_csv(filename, mode='a', index=False, header=False)


clean_data()
vectorize()
test_review()
test_data()
