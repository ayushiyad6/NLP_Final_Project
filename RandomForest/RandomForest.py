import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import re
import nltk
from nltk.tokenize import RegexpTokenizer
import re
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer

stop_words = set(stopwords.words('english'))

stemmer = PorterStemmer()

def pre_process(dataset):
    global stop_words
    pattern = RegexpTokenizer(r'\w+')
    pattern2 = re.compile('[0-9]+')

    new_list = []

    for i in range(0,len(dataset)):
        temp = dataset[i]
        if '\n' in temp:  # Any new lines present in the same column remove it
            temp = temp.replace('\n', " ")
        if r'<br />' in temp:
            temp = temp.replace(r'<br />', " ")
        temp_word_list =  pattern.tokenize(temp)
        s = ""
        temp_set = set()
        result = []
        for j in temp_word_list:
            j = j.lower()
            if re.search(pattern2, j):
                pass
            else:
                if j not in stop_words:
                    if j not in temp_set:
                        j= stemmer.stem(j)
                        temp_set.add(j)
                        result.append((j))

        # print(result)
        for k in result:
            s = s + " " + k
        new_list.append(word_tokenize(s))

    return new_list

train_set =pd.read_csv(r'C:\Users\epava\OneDrive\Documents\NLP\Project\Train\Train.csv')
test_set =pd.read_csv(r'C:\Users\epava\OneDrive\Documents\NLP\Project\Test\Test.csv')
train_set['new_text'] = pre_process(train_set['text'])
test_set['new_text'] = pre_process(test_set['text'])

train = train_set['new_text'].apply(lambda x: ' '.join(x))
train_label = train_set['label']
test = test_set['new_text'].apply(lambda x: ' '.join(x))
test_label = test_set['label']

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
count_vect = CountVectorizer(stop_words='english')
transformer = TfidfTransformer(norm='l2',sublinear_tf=True)

x_train_counts = count_vect.fit_transform(train)
x_train_tfidf = transformer.fit_transform(x_train_counts)
#print(x_train_counts.shape)
#print(x_train_tfidf.shape)

#Output :(25569, 27304) (25569, 27304)
x_test_counts = count_vect.transform(test)
x_test_tfidf = transformer.transform(x_test_counts)
#print(x_test_counts.shape)
#print(x_test_tfidf.shape)
#Output : (6393, 27304) (6393, 27304)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=200)
model.fit(x_train_tfidf,train_label)
predictions = model.predict(x_test_tfidf)

#Confusion Matrix
from sklearn.metrics import confusion_matrix,f1_score
CM = confusion_matrix(test_label,predictions)
print(CM)

#f1-score
F1 = f1_score(test_label,predictions)
print("F1-Score : ", F1)


#Accuracy_score
from sklearn.metrics import accuracy_score
Acc = accuracy_score(test_label,predictions)*100
print("Accuracy : ", Acc)

