'''
Author: Ayushi Yadav 
        Nandan Prasad
        Sai Pavan

Date: 30th March 2023

Title : NLP_Project_NB.py

Description : The python code trains the dataset of Naive Bayes model, 
            tests the data and gives the score of the model that is achieved.

Functions: It has 5 functions:
            1. Create_Test_Dataset(TestFile, TestFileLabel)
                The test dataset contains 0, 1 and -1 labels. 
                Filtering the data with only 0 and 1 labels.
            2. clean_train_tweets(df)
                Cleaning the data
            3. clean_data()
                saves the clean data in a file
            4. train_NB_model(path_to_train_file)
                training the model with the clean train data
            5. test_NB_model(path_to_test_file, NB_model)
                testing the test data on file created in step 1.
            6. Model_score(path_to_test_file, NB_model)
                Gives the score of the model.

'''


import pandas as pd
import random
import re
import numpy as np
import csv
import preprocessor as p
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
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


global tempArr
tempArr = []

global vec

replace_no_space = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\|)|(\()|(\))|(\[)|(\])|(/%)|(\$)|(\>)|(\<)|(\{)|(\})")
replace_with_space = re.compile("(<br\s/><br\s/?)|(-)|(/)|(:).")



def clean_train_tweets(df):

    # clearing the tempArr list if previous entries exist

    global tempArr
    tempArr.clear()

    for i in range(len(df)):
        tmpl = p.clean(df[i])                                 # preprocessor library to clean data
        if '\n' in tmpl:                                    # Any new lines present in the same column remove it
            tmpl = tmpl.replace('\n', " ")
        if r'<br />' in tmpl:
            tmpl = tmpl.replace(r'<br />', " ")
        # tmpl = replace_no_space.sub("", tmpl.lower())        # lower casing
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
        tempArr.append(filtered_sentence)                   # appending it to gloabl array
    return tempArr



def clean_data():

    # Creating the file and the header of the csv file.

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
    
    # Populating the csv file

    for i in range(len(tempArr)):
        x = random.randint(0,1000000)
        data_to_append = ({ 'text' : [tempArr[i]], 'label' : [df.label[i]]})
        df2 = pd.DataFrame(data_to_append)
 
        # append data frame to CSV file
        df2.to_csv(filename, mode='a', index=False, header=False)



def train_NB_model(path_to_train_file):
    
    # segregating the data x_train and y_train_model

    df_train = pd.read_csv(path_to_train_file)
    x_train = df_train.text.fillna(' ')
    y_train_label = df_train['label']

    # creating vectorizer object

    vectorizer = CountVectorizer()
    x_train_count = vectorizer.fit_transform(x_train)

    global vec
    vec = vectorizer

    # Training the model

    model = MultinomialNB()
    model.fit(x_train_count, y_train_label)
    return model





def test_NB_model(path_to_test_file, NB_model):

    # Reading the testing file path

    test = pd.read_csv(path_to_test_file)
    testArray = clean_train_tweets(test["text"])

    # creating an empty list of score of each and every entry
    testArrayscore = []

    # vectorizer = CountVectorizer()

    # populating the list
    for item in testArray:
        cmt = vec.transform(np.array([item]))
        result = NB_model.predict(cmt)
        testArrayscore.append(result[0])

    # creating file
    headerList = ['text', 'label']
    filename = "final_result.csv"
    df = pd.read_csv("train.csv")
    with open(filename, 'w') as file:
        dw = csv.DictWriter(file, delimiter=',', 
                        fieldnames=headerList)
        dw.writeheader()
    df = pd.read_csv("Test.csv")

    # Populating the csv file with the tweet and the score

    for i in range(len(testArrayscore)):
        data_to_append = ({ 'text' : [df["text"][i]], 'label' : [testArrayscore[i]]})
        df2 = pd.DataFrame(data_to_append)
 
        # append data frame to CSV file
        df2.to_csv(filename, mode='a', index=False, header=False)


def Model_score(path_to_test_file, NB_model):

    # vectorizer = CountVectorizer()
    test = pd.read_csv(path_to_test_file)
    x_test = test.text
    y_test = test.label
    x_test_count = vec.transform(x_test)
    model_score = NB_model.score(x_test_count, y_test)
    print("The model score is: ", model_score)




# Calling the various functions

clean_data()

NB_model = train_NB_model("clean_train.csv")

test_NB_model("Test.csv", NB_model)

Model_score("Test.csv", NB_model)

# Code ends