###### IMPORTS FOR IMPORT AND AUDIO NORMALIZATION
from scipy.io import wavfile
from os import listdir
from os.path import isfile, join
import os.path
import pathlib
################################

from scipy.io import wavfile as wav
from python_speech_features import mfcc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import keras


################################ IMPORT MODULE ################################

audio_path = 'data'
normalized_audio_path = 'normalized_data'
test_audio_path = 'test_normalized_data'
pathlib.Path(normalized_audio_path).mkdir(parents=True, exist_ok=True)

def get_number (audio_name):
    number_name = audio_name [0 : audio_name.find('_')]
    num_array = ['cero', 'uno', 'dos', 'tres', 'cuatro', 'cinco', 'seis', 'siete', 'ocho', 'nueve', 'diez',
                    'once', 'doce', 'trece', 'catorce', 'quince']
    num = num_array.index(number_name)
    return num

#Load all the audios and normalize them
#Returns a list with the normalized audios and save the normalized audios in the normalized_audios directory.
def load_audios (audio_path):
    normalized_audios = []
    normalized_audio_files = [f for f in listdir(audio_path) if isfile(join(audio_path, f))]
    for audio_name in normalized_audio_files:
        fs, normalized_audio = wav.read(audio_path + '/' + audio_name)
        normalized_audios.append ([normalized_audio, get_number(audio_name)])
    return fs, normalized_audios
################################################################################


#Rerurns the MFCC vector and the number
def get_training_set(numcep, numcontext):
    training_set = []
    label_set = []
    fs, normalized_audios = load_audios(normalized_audio_path)
    for i in range (len(normalized_audios)):
        audio = normalized_audios [i][0]
        number = normalized_audios [i][1]
        audio_info = input_vector(audio, fs, numcep, numcontext)
        #audio_info=np.reshape(100,audio_info.shape[1])
        label_set.append(number)
        training_set.append(audio_info)
    assert(len(label_set) == len(training_set))
    return np.asarray(training_set), np.asarray(label_set)
    

#Rerurns the MFCC vector and the number
def get_test_set(numcep, numcontext):
    training_set = []
    label_set = []
    fs, normalized_audios = load_audios(test_audio_path)
    for i in range (len(normalized_audios)):
        audio = normalized_audios [i][0]
        number = normalized_audios [i][1]
        audio_info = input_vector(audio, fs, numcep, numcontext)
        #audio_info=np.reshape(100,audio_info.shape[1])
        label_set.append(number)
        training_set.append(audio_info)
    assert(len(label_set) == len(training_set))
    return np.asarray(training_set), np.asarray(label_set)

def input_vector(audio, fs, numcep, numcontext):
    '''
    Turn an audio file into feature representation.
    This function has been modified from Mozilla DeepSpeech:
    https://github.com/mozilla/DeepSpeech/blob/master/util/audio.py
    and
    https://github.com/mrubash1/RNN-Tutorial/blob/master/src/features/utils/load_audio_to_mem.py
    # This Source Code Form is subject to the terms of the Mozilla Public
    # License, v. 2.0. If a copy of the MPL was not distributed with this
    # file, You can obtain one at http://mozilla.org/MPL/2.0/.
    '''
    
    # Get mfcc coefficients
    orig_inputs = mfcc(audio, samplerate=fs, numcep=numcep)
    # We only keep every second feature (BiRNN stride = 2) ???
    orig_inputs = orig_inputs[::2]
    # For each time slice of the training set, we need to copy the context this makes
    # the numcep dimensions vector into a numcep + 2*numcep*numcontext dimensions
    # because of:
    #  - numcep dimensions for the current mfcc feature set
    #  - numcontext*numcep dimensions for each of the past and future (x2) mfcc feature set
    # => so numcep + 2*numcontext*numcep
    train_inputs = np.array([], np.float32)
    train_inputs.resize(orig_inputs.shape[0], numcep + 2 * numcep * numcontext)
    # Prepare pre-fix post fix context
    empty_mfcc = np.array([])
    empty_mfcc.resize((numcep))
    # Prepare train_inputs with past and future contexts
    time_slices = range(train_inputs.shape[0])
    context_past_min = time_slices[0] + numcontext
    context_future_max = time_slices[-1] - numcontext
    for time_slice in time_slices:
        # Reminder: array[start:stop:step]
        # slices from indice |start| up to |stop| (not included), every |step|

        # Add empty context data of the correct size to the start and end
        # of the MFCC feature matrix

        # Pick up to numcontext time slices in the past, and complete with empty
        # mfcc features
        need_empty_past = max(0, (context_past_min - time_slice))
        empty_source_past = list(empty_mfcc for empty_slots in range(need_empty_past))
        data_source_past = orig_inputs[max(0, time_slice - numcontext):time_slice]
        assert(len(empty_source_past) + len(data_source_past) == numcontext)

        # Pick up to numcontext time slices in the future, and complete with empty
        # mfcc features
        need_empty_future = max(0, (time_slice - context_future_max))
        empty_source_future = list(empty_mfcc for empty_slots in range(need_empty_future))
        data_source_future = orig_inputs[time_slice + 1:time_slice + numcontext + 1]
        assert(len(empty_source_future) + len(data_source_future) == numcontext)

        if need_empty_past:
            past = np.concatenate((empty_source_past, data_source_past))
        else:
            past = data_source_past

        if need_empty_future:
            future = np.concatenate((data_source_future, empty_source_future))
        else:
            future = data_source_future

        past = np.reshape(past, numcontext * numcep)
        now = orig_inputs[time_slice]
        future = np.reshape(future, numcontext * numcep)

        train_inputs[time_slice] = np.concatenate((past, now, future))
        assert(len(train_inputs[time_slice]) == numcep + 2 * numcep * numcontext)

    # Scale/standardize the inputs
    # This can be done more efficiently in the TensorFlow graph
    train_inputs = (train_inputs - np.mean(train_inputs)) / np.std(train_inputs)
    return train_inputs


def main():
    X, Y = get_training_set(13, 9)
    print(X.shape)
    print(Y.shape)
    X = keras.preprocessing.sequence.pad_sequences(X, maxlen=100)
    print(X.shape)


if __name__== "__main__":
  main()
