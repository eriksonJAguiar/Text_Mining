import re
import nltk
import sys
import numpy as np
import pandas as pd
import statistics
from nltk.stem.wordnet import WordNetLemmatizer
from unicodedata import normalize
from pyfasttext import FastText

#Modelos de classificacao
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import svm

#Metricas
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix

#Outros Sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict, KFold,GroupKFold
from sklearn.preprocessing import label_binarize
from sklearn.feature_extraction.text import TfidfTransformer

args = sys.argv
dataset_name = args[1]
#output = args[2]


def embeddings(data):

    model = FastText('skipModelNilc.bin')

    word_vec = []
    for token in data:
        vec = model[token].tolist()
        word_vec.append(vec)
    
    print("Word Embeddings concluido")

    return word_vec


def preprocessing(data):


    #limpeza do texto
    data = re.sub(r"@ \w", "", data)
    data = re.sub(r"# \w", "", data)
    data = re.sub(r"@ \d", "", data)
    data = re.sub(r"# \d", "", data)
    data = re.sub(r"[^\w\d\s#@]", "", data)
            
    #extracao do tokens e remocao stop words
    tokens = [w for w in nltk.regexp_tokenize(data.lower(),"[\S]+") if not w in nltk.corpus.stopwords.words('portuguese') and len(w) > 1]

    new_data = []

    #lematizacao
    lm = WordNetLemmatizer()

    for token in tokens:

        lema = lm.lemmatize(token)

        new_data.append(lema)


    print("Pre-processamento concluido")
    

    return new_data

def classify(model, train, target):

        count_vect = CountVectorizer()
        X = count_vect.fit_transform(train)
        kf = KFold(10, shuffle=True, random_state=1)

        ac_v = []
        cm_v = []
        p_v = []
        r_v = []
        f1_v = []
        e_v = []
        fpr = []
        tpr = []
        roc_auc_ = []
        predicts = []


        for train_index,teste_index in kf.split(X,target):
            X_train, X_test = X[train_index],X[teste_index]
            y_train, y_test = target[train_index], target[teste_index]
            model.fit(X_train,y_train)
            pred = model.predict(X_test)
            ac = accuracy_score(y_test, pred)
            p = precision_score(y_test, pred,average='weighted')
            r = recall_score(y_test, pred,average='weighted')
            f1 = (2*p*r)/(p+r)
            e = mean_squared_error(y_test, pred)
            cm = confusion_matrix(y_test,pred)
            cm_v.append(cm)
            ac_v.append(ac)
            p_v.append(p)
            r_v.append(r)
            f1_v.append(f1)
            e_v.append(e)


        ac = statistics.median(ac_v)
        p = statistics.median(p_v)
        f1 = statistics.median(f1_v)
        r = statistics.median(r_v)
        e = statistics.median(e_v)
        
        return predicts,ac,ac_v,p,r,f1,e

if __name__ == '__main__':

    #dataset = open(dataset_name,'r', encoding="utf-8")

    dataset = pd.read_csv(dataset_name, sep=';', encoding='utf-8')

    print(dataset.head())

    #arquivo = open(output,'w', encoding="utf-8")

    new_dataset = []
    
        
    for line in dataset:

        data = line.rstrip()

        pre_data = preprocessing(data)

    
    dataset.close()