
#fonte: https://www.kaggle.com/pierremegret/dialogue-lines-of-the-simpsons

import re
import nltk
from nltk.stem import PorterStemmer
from sklearn import preprocessing
#nltk.download('rslp')
#nltk.download('stopwords')
import numpy as np
import pandas as pd
from unicodedata import normalize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans


def loadDataset(path):

    datas = pd.read_csv(path,sep=",")

    datas.columns = ['person','text']

    return datas


def preProcessing(dataset):

    stemmer = PorterStemmer()
        
    datasetNorm = []
    
    for row in dataset:
        print(row)
        expr = re.sub(r"[^\w\d\s]", "", row)
        expr = normalize('NFKD',expr).encode('ASCII','ignore').decode('ASCII')
        filt = [w for w in nltk.regexp_tokenize(expr.lower(),"[\S]+") if not w in nltk.corpus.stopwords.words('english')]
        
        sentense = ""
        
        for f in filt:
            f = stemmer.stem(f)
            sentense += f + " "
        
        datasetNorm.append(sentense)

    

    return datasetNorm


def prePerson(dataset):
    
    le = preprocessing.LabelEncoder()

    p = le.fit(dataset)

    return p




def datasetNorm(dataset):

   
   #d = prePerson(dataset['person'].tolist())
   #print(d)
   dataset['text'] = preProcessing(dataset['text'])
   
    
   vectorizer = CountVectorizer()
   
   dataset['text'] = vectorizer.fit_transform(dataset['text'])
   
   kmeans = KMeans(n_clusters=10, random_state=0).fit(dataset.values)

   return kmeans.cluster_centers_


dataset = loadDataset("../simpsons_dataset.csv")
dataset = dataset.head(100)
#d  = dataset['title'].tolist()[1:100]
#d = preProcessing(d)
centers = datasetNorm(dataset)
print(centers)
