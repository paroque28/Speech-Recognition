#!/bin/python3
import numpy as np
import keras
import sys
import matplotlib.pyplot as plt
from keras.models import load_model
from dataset import input_vector, get_test_set
from scipy.io import wavfile as wav
from plots import plot_confusion_matrix
from sklearn.metrics import confusion_matrix



Classes = ['cero','uno','dos','tres','cuatro','cinco','seis','siete','ocho','nueve','diez','once','doce','trece','catorce', 'quince']

def predictNumber(audioPath):
    model=load_model('model/model4.hdf5')

    ##procesar audio, o lo que sea y meter en audio
    fs, audio = wav.read(audioPath) ## tiene que ser normalizado
    X = input_vector(audio, fs, 13, 9)

    
    X=X.reshape(1,X.shape[0],X.shape[1])
    X = keras.preprocessing.sequence.pad_sequences(X, maxlen=100)

    number=model.predict(X)
    print("System Output: " + str(Classes[np.argmax(number)]))




def plot():
    model=load_model('model/model4.hdf5')

    x_test, y_test=get_test_set(13,9)
    x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=100)

    y_predict = model.predict_classes (x_test, verbose=0)
    
    plt.figure()
    cm = confusion_matrix(y_test,y_predict)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plot_confusion_matrix(cm, classes=Classes, title='Confusion matrix')
    plt.show()



if __name__=='__main__':
    type=sys.argv[1]

    if type=="-p":
        audio=sys.argv[2]
        predictNumber(audio)
    elif type=="-m":
        plot()