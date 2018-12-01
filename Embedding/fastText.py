from pyfasttext import FastText
import numpy as np

model = FastText('skipModelNilc.bin')

sentence = ['hoje', 'acordei', 'puto']

if __name__ == '__main__':

    word_vec = []
    for s in sentence:
        vec = model[s].tolist()
        word_vec.append(vec)
        #print(vec)



    npw = np.array(word_vec)
    print(npw)
