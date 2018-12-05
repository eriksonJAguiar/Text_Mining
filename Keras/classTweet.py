from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
import numpy as np

train = np.loadtxt("TrainDatasetFinal.txt", delimiter=",")
test = np.loadtxt("testDatasetFinal.txt", delimiter=",")

y_train = train[:,7]
y_test = test[:,7]

train_spec = train[:,6]
test_spec = test[:,6]


model = Sequential()
model.add(LSTM(32, input_shape=(1415684, 8)))
model.add(LSTM(64, input_dim=1, input_length=1415684, return_sequences=True))

model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop')

model.fit(train_spec, y_train, batch_size=2000, nb_epoch=11)
score = model.evaluate(test_spec, y_test, batch_size=2000)
