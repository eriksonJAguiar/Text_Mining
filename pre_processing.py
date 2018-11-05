
#fonte: https://www.kaggle.com/marlesson/news-of-the-site-folhauol/version/1

import re
import nltk
#nltk.download('rslp')
#nltk.download('stopwords')
import numpy as np
import pandas as pd
from unicodedata import normalize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans


def loadDataset(path):

    datas = pd.read_csv(path,sep=",")

    return datas


def preProcessing(dataset):

    stemmer = nltk.stem.RSLPStemmer()
        
    datasetNorm = []
    
    for row in dataset:
        expr = re.sub(r"[^\w\d\s]", "", row)
        expr = normalize('NFKD',expr).encode('ASCII','ignore').decode('ASCII')
        filt = [w for w in nltk.regexp_tokenize(expr.lower(),"[\S]+") if not w in nltk.corpus.stopwords.words('portuguese')]
        
        sentense = ""
        
        for f in filt:
            f = stemmer.stem(f)
            sentense += f + " "
        
        datasetNorm.append(sentense)

    

    return datasetNorm


def preProcessingData(dataset):

    new = []
    for s in dataset:
        s = s.replace('-','')
        new.append(int(s))

    return new




def datasetNorm(dataset):

   
   dataset['title'] = preProcessing(dataset['title'])
   dataset['text'] = preProcessing(dataset['text'])
   dataset['date'] = preProcessingData(dataset['date'])

    
   vectorizer = CountVectorizer()
   
   dataset['title'] = vectorizer.fit_transform(dataset['title'])
   dataset['text'] = vectorizer.fit_transform(dataset['text'])
   
   kmeans = KMeans(n_clusters=10, random_state=0).fit(dataset.values)

   return kmeans.cluster_centers_


dataset = loadDataset("../articles.csv")
dataset = dataset.head(100)
#d  = dataset['title'].tolist()[1:100]
#d = preProcessing(d)
centers = datasetNorm(dataset)
print(centers)
