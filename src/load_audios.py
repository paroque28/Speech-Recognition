from scipy.io import wavfile
from os import listdir
from os.path import isfile, join
import os.path
from pydub import AudioSegment
from pydub.playback import play #PRUEBAS sirve para reproducir audios

import pathlib

audio_path = 'data'
normalized_audio_path = 'normalized_data'
test_audio_path = 'test_data'
test_normalized_audio_path = 'test_normalized_data'
pathlib.Path(normalized_audio_path).mkdir(parents=True, exist_ok=True)


normalized_audios = []

def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)



def normalize(audio_path, normalized_audio_path):
    audio_files = [f for f in listdir(audio_path) if isfile(join(audio_path, f))]
    for audio_name in audio_files:
        sound = AudioSegment.from_file(audio_path + '/' + audio_name)
        normalized_audio = match_target_amplitude(sound, -20)
        normalized_audio.export (normalized_audio_path + '/' + audio_name, format = 'wav')
        #normalized_audios.append (normalized_audio)

    

if __name__== "__main__":
    normalize(audio_path,normalized_audio_path)
    normalize(test_audio_path,test_normalized_audio_path)


#fs, data = wavfile.read(filename)
