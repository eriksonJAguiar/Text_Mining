import re
import nltk
import sys
import numpy as np
from nltk.stem.wordnet import WordNetLemmatizer
from unicodedata import normalize
from pyfasttext import FastText

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


if __name__ == '__main__':

    dataset = open(dataset_name,'r', encoding="utf-8")

    #arquivo = open(output,'w', encoding="utf-8")

    new_dataset = []
    
        
    for line in dataset:

        data = line.rstrip()

        pre_data = preprocessing(data)

    
    dataset.close()