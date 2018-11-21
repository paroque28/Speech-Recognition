import numpy as np
import keras
import sys
import matplotlib.pyplot as plt
from keras.models import load_model
from process import input_vector
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
    plt.figure()
    model=load_model('model/model4.hdf5')

    y_data=np.arange(16)
    x_data=[]

    for number in Classes:
        fs, audio = wav.read("test_data/"+str(number)+"_javier.wav") ## tiene que ser normalizado
        data = input_vector(audio, fs, 13, 9)
        data =data.reshape(1,data.shape[0],data.shape[1])
        data = keras.preprocessing.sequence.pad_sequences(data, maxlen=100)
        data = data.reshape(data.shape[1],data.shape[2])
        x_data.append(data)
    x_data = np.asarray(x_data)

    x_predict = model.predict_classes(x_data,verbose=0)

    confMatrix = confusion_matrix(x_predict,y_data)
    print(confMatrix)
    plot_confusion_matrix(confMatrix, classes=Classes, title='Confusion matrix')
    plt.show()



if __name__=='__main__':
    type=sys.argv[1]

    if type=="-p":
        audio=sys.argv[2]
        predictNumber(audio)
    elif type=="-m":
        plot()