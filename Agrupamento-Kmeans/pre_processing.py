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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import Counter


def loadDataset(path):

    datas = pd.read_csv(path,sep=",")

    datas.columns = ['person','text']

    return datas


def preProcessing(dataset):

    stemmer = PorterStemmer()
        
    datasetNorm = []
    
    for row in dataset:
        expr = re.sub(r"[^\w\d\s]", "", row)
        expr = normalize('NFKD',expr).encode('ASCII','ignore').decode('ASCII')
        filt = [w for w in nltk.regexp_tokenize(expr.lower(),"[\S]+") if not w in nltk.corpus.stopwords.words('english')]
        
        sentense = ""
        
        for f in filt:
            f = stemmer.stem(f)
            sentense += f + " "
        
        datasetNorm.append(sentense)       

    

    return datasetNorm


def bow(corpus):

    bag = []
    
    for c in corpus:

        aux = c.split(' ')
        bag = bag + aux
    
    count_bag = Counter(bag)
    
    bow = []

    mx = len(max(corpus))
    
    for s in corpus:
        aux = []
        vet = s.split(' ')
        for i in range(mx):
            if i < len(vet): 
             aux.append(count_bag[vet[i]])
            else:
              aux.append(0)
        
        bow.append(np.array(aux))
    

    return bow



def prePerson(dataset):
    
    le = preprocessing.LabelEncoder()

    p = le.fit_transform(dataset)

    return p


def plot_SSE(data):

    sse = {}
   
    for k in range(1, 10):
       kmeans = KMeans(n_clusters=k,init='k-means++', max_iter=1000).fit(data)
       sse[k] = kmeans.inertia_ 

    plt.figure()
    plt.plot(list(sse.keys()), list(sse.values()))
    plt.xlabel("Number of cluster")
    plt.ylabel("SSE")
    plt.show()

    return kmeans.cluster_centers_, kmeans.labels_


def datasetNorm(dataset):

   df = pd.DataFrame(columns=['person','text'])

   #df['person'] = prePerson(dataset['person'])
   #print('pre-processando o texto ...')
   #df['text'] = preProcessing(dataset['text'])
   #print('pre-processando pronto !')
   
   print('transformando o texto ..')
   #df['text'] = bow(df['text'])
   vectorizer = TfidfVectorizer(stop_words='english')
   X = vectorizer.fit_transform(dataset['text'].tolist())

   print('agrupando ...')
   #centers, labels = plot_SSE(X)
   
   kmeans = KMeans(n_clusters=8,init='k-means++', max_iter=1000).fit(X)
   y_kmeans = kmeans.predict(X)

   centers = kmeans.cluster_centers_
   plt.scatter(X[:,0],X[:,1], s=50, cmap='simpson')
   plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)

   #kmeans.cluster_centers_, kmeans.labels_
   #centers, labels

   return kmeans.cluster_centers_, kmeans.labels_


print('iniciando o algoritmos...')
dataset = loadDataset("../simpsons_dataset.csv")
dataset = dataset.dropna()
#dataset = dataset.head(100)
centers,labels = datasetNorm(dataset)
dataset['label'] = labels
#dataset.to_csv('simpson_gruping.csv', sep=';', encoding='utf-8')
