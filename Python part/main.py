import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk

nltk.download('stopwords')
print(stopwords.words('english'))

#Data pre-processing

news_dataset = pd.read_csv('D:/projects/Fake news prediction/Python part/fake_real_news_dataset.csv')
#print(news_dataset.shape)
#print(news_dataset.head())

#print(news_dataset.isnull().sum())
news_dataset = news_dataset.fillna('')

X = news_dataset.drop(columns='label', axis=1)

Y = news_dataset['label']

port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content).lower().split()
    stemmed_content = ' '.join([port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')])
    return stemmed_content

news_dataset['title'] = news_dataset['title'].apply(stemming)
news_dataset['author'] = news_dataset['author'].apply(stemming)

news_dataset['content'] = news_dataset['author']+' '+news_dataset['title']

X = news_dataset['content'].values
Y = news_dataset['label'].values

vectorizer = TfidfVectorizer()
vectorizer.fit(X)

X = vectorizer.transform(X)


#print(X)


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, stratify = Y, random_state=2)

model = LogisticRegression()
model.fit(X_train,Y_train)

X_train_prediction = model.predict(X_train)
training_accuracy = accuracy_score(X_train_prediction,Y_train)

X_test_prediction = model.predict(X_test)
training_accuracy_test = accuracy_score(X_test_prediction,Y_test)

print("Accuracy of trained: ", training_accuracy)
print("Accuracy of tested: ", training_accuracy_test)

