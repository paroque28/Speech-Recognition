import numpy as np
from keras.models import load_model
import sys


def predictNumber():
    model=load_model('models/model.hdf5')

    ##procesar audio, o lo que sea y meter en audio
    audio= "shit "
    number=model.predict_clases(audio)

    print (number)
