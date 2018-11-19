
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout , Activation
from keras.layers import LSTM
from process import get_training_set
from model import getModel

def train():
    clases=16
    imputShape=(1,1,1)
    x_train,y_train=get_training_set()
    x_test=x_train[1000:]
    y_test=y_train[1000:]
    x_train=x_train[:1000]
    y_train=y_train[:1000]


    y_train = keras.utils.to_categorical(y_train, 16)
    y_test = keras.utils.to_categorical(y_test, 16)

    model = getModel(imputShape, clases)

    model.fit(x_train, y_train,batch_size=batch_size,epochs=20, verbose=1,validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    model.save('models/model.hdf5')

    print(model.summary())


