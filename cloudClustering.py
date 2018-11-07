from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd 
import re
from unicodedata import normalize 
import nltk
from collections import Counter

def preProcessing(dataset):
    
    datasetNorm = []
    
    for row in dataset: 
        expr = re.sub(r"[^\w\d\s]", "", row)
        expr = normalize('NFKD',expr).encode('ASCII','ignore').decode('ASCII')
        filt = [w for w in nltk.regexp_tokenize(expr.lower(),"[\S]+") if not w in nltk.corpus.stopwords.words('english')]

        sent = ""

        for s in filt:
            sent = sent +" "+s

        
        datasetNorm.append(sent)


    return datasetNorm


def freq(corpus):

    bag = []

    for c in corpus:

        aux = c.split(' ')
        bag = bag + aux

    count_bag = Counter(bag)

    return count_bag



#text = open('debate.csv','r').read()

def cloud(datas):

    datas['text'] = preProcessing(datas['text'].tolist())

    
    for i in range(0,8):
        #df.loc[lambda df: df['label'] == 1]
        df = datas.loc[datas['label'] == str(i)]
        text = freq(df['text'].tolist())
        wc = WordCloud(background_color="white", max_words=50).generate_from_frequencies(text)
        #wordcloud = WordCloud(max_font_size=100,width = 1520, height = 535).generate(text)
        f = plt.figure(figsize=(16,9))
        plt.imshow(wc)
        #plt.save('imgs_%i'%(i), wordcloud)
        plt.axis("off")
        plt.show()
        f.savefig('imgs_%i'%(i))


dataset = pd.read_csv('simpson_gruping.csv', sep=";")
dataset.columns = ['person','text','label']
dataset = dataset.dropna()
#dataset = dataset.head(100)
cloud(dataset)
