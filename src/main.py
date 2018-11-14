from scipy.io import wavfile as wav
from python_speech_features import mfcc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def readWav(audio_filename, numcep):
    # Load wav files
    fs, audio = wav.read(audio_filename)
    
    # Get mfcc coefficients
    orig_inputs = mfcc(audio, samplerate=fs, numcep=numcep)
    
    ig, ax = plt.subplots()
    mfcc_data= np.swapaxes(orig_inputs, 0 ,1)
    cax = ax.imshow(mfcc_data, interpolation='nearest', cmap=cm.jet, origin='lower', aspect='auto')
    ax.set_title('MFCC')
    #Showing mfcc_data
    plt.show()
    #Showing orig_inputs
    plt.plot(orig_inputs)
    plt.show()
    # For each time slice of the training set, we need to copy the context this makes
    train_inputs = np.array([], np.float32)
    train_inputs.resize((orig_inputs.shape[0], numcep + 2 * numcep * numcontext))
    
    for time_slice in range(train_inputs.shape[0]):
        # Pick up to numcontext time slices in the past,
        # And complete with empty mfcc features
        need_empty_past = max(0, ((time_slices[0] + numcontext) - time_slice))
        empty_source_past = list(empty_mfcc for empty_slots in range(need_empty_past))
        data_source_past = orig_inputs[max(0, time_slice - numcontext):time_slice]
        assert(len(empty_source_past) + len(data_source_past) == numcontext)


readWav("../data/catorce_0_2.wav" , 13)