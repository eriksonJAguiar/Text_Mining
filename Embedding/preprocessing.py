import re
import nltk
import sys
from nltk.stem.wordnet import WordNetLemmatizer
from unicodedata import normalize
#nltk.download('rslp')
#nltk.download('stopwords')
  

args = sys.argv
dataset = args[1]
output = args[2]

if __name__ == '__main__':

    
    dataset = open(dataset,'r', encoding="utf-8")

    arquivo = open(output,'w', encoding="utf-8")
    
        
    for line in dataset:
        
        data = line.rstrip()
        
        #limpeza do texto
        data = re.sub(r"[^\w\d\s]", "", data)
            
        #extracao do tokens e remocao stop words
        tokens = [w for w in nltk.regexp_tokenize(data.lower(),"[\S]+") if not w in nltk.corpus.stopwords.words('portuguese') and len(w) > 1]

        new_data = ""

        #lematizacao
        lm = WordNetLemmatizer()

        for token in tokens:

            lema = lm.lemmatize(token)

            new_data = new_data + " " + lema

        
        arquivo.write(new_data + "\n")


    print("Operacao concluida no arquivo "+  arquivo.name + " de pre-processamento")
    arquivo.close()




        




            
