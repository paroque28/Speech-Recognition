
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout , Activation
from keras.layers import LSTM
from main import get_training_set



x_train,y_train=get_training_set()
x_test=x_train[1000:]
y_test=y_train[1000:]
x_train=x_train[:1000]
y_train=y_train[:1000]
print(x_train.shape())


model = Sequential()
model.add(LSTM(16, input_shape,return_sequences=True))
model.add(Dropout(0.3))
model.add(Dense(16, activation='softmax'))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())