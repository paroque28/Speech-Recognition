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
    model=load_model('model/model.hdf5')

    ##procesar audio, o lo que sea y meter en audio
    fs, audio = wav.read(audioPath) ## tiene que ser normalizado
    X = input_vector(audio, fs, 13, 9)

    
    X=X.reshape(1,X.shape[0],X.shape[1])
    X = keras.preprocessing.sequence.pad_sequences(X, maxlen=100)



    number=model.predict(X)
    print(Classes[np.argmax(number)])




# def plot():
#     plt.figure()
#     model=load_model('model/model.hdf5')

#     b=[]
#     a=[]

    


#     for x in range(0,15):
#         a.insert(x,y_train[x])
#         b.insert(x,model.predict_classes(x_train[x],verbose=0)[0])

#     confMatrix = confusion_matrix(b,a)
#     plot_confusion_matrix(confMatrix, classes=Classes, title='confusion matrix')
#     plt.show()



if __name__=='__main__':
    audio=sys.argv[1]
    predictNumber(audio)