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
print(news_dataset.shape)
print(news_dataset.head())

print(news_dataset.isnull().sum())
news_dataset = news_dataset.fillna('')

X = news_dataset.drop(columns='label', axis=1)

Y = news_dataset['label']

port_stem = PorterStemmer()
