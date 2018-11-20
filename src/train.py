
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout , Activation
from keras.layers import LSTM
from process import get_training_set
from model import getModel


Classes = ['cero','uno','dos','tres','cuatro','cinco','seis','siete','ocho','nueve','diez','once','doce','trece','catorce', 'quince']

def train():
    clases=16
    inputShape=(100,247)
    x_train,y_train=get_training_set(13,9)
    x_test=x_train[1000:]
    y_test=y_train[1000:]
    x_train=x_train[:1000]
    y_train=y_train[:1000]

    x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=100)
    x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=100)


    y_train = keras.utils.to_categorical(y_train, 16)
    y_test = keras.utils.to_categorical(y_test, 16)
    print(y_train)
    print(x_train.shape)

    
    model = getModel(inputShape, clases)

    model.fit(x_train, y_train,batch_size=10,epochs=100, verbose=1,validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    model.save('model/model2.hdf5')



if __name__=='__main__':
    train()