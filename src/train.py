
import numpy as np
import keras
from sklearn.model_selection import train_test_split
from dataset import get_training_set, get_test_set
from model import getModel
from plots import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def train():
    # Training set
    x_train, y_train = get_training_set(13,9)
    x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=100)
    y_train = keras.utils.to_categorical(y_train, 16)
    # Test set
    x_test, y_test=get_test_set(13,9)
    x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=100)
    y_test = keras.utils.to_categorical(y_test, 16)
    
    
    model = getModel((x_train.shape[1],x_train.shape[2]), y_train.shape[1])

    model.fit(x_train, y_train,batch_size=10,epochs=104, verbose=1,validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    model.save('model/model5.hdf5')

if __name__=='__main__':
    train()