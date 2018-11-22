
import numpy as np
import keras
from sklearn.model_selection import train_test_split
from process import get_training_set
from model import getModel
from plots import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def train():
    x_data, y_data=get_training_set(13,9)
    x_data = keras.preprocessing.sequence.pad_sequences(x_data, maxlen=100)
    
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data)
    y_train = keras.utils.to_categorical(y_train, 16)
    y_test_ = keras.utils.to_categorical(y_test, 16)
    
    
    model = getModel((x_train.shape[1],x_train.shape[2]), y_train.shape[1])

    model.fit(x_train, y_train,batch_size=10,epochs=104, verbose=1,validation_data=(x_test, y_test_))
    score = model.evaluate(x_test, y_test_, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    model.save('model/model5.hdf5')

    y_predict = model.predict_classes (x_test, verbose=0)
    #y_predict = keras.utils.to_categorical(y_predict, 16)
    Classes = ['cero','uno','dos','tres','cuatro','cinco','seis','siete','ocho','nueve','diez','once','doce','trece','catorce', 'quince']
    plt.figure()
    confMatrix = confusion_matrix(y_test,y_predict)
    print(confMatrix)
    plot_confusion_matrix(confMatrix, classes=Classes, title='Confusion matrix')
    plt.show()

if __name__=='__main__':
    train()