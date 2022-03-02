# This script is to load, manupipulate and store our audio data.

import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

four_on_floor_titles = librosa.util.find_files(r'C:\Users\chris\Desktop\ECEC487Materials\RecognitionProject\songs\Vocal Pieces',ext=None,recurse=False,case_sensitive=False,limit=1,offset=0)

four_on_floor_stft = []
for title in four_on_floor_titles:

    y, sr = librosa.load(title,sr=12000,mono=True)
    y, index = librosa.effects.trim(y)

    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    y = y[beats[17]:]
    y = librosa.util.fix_length(y,sr*15)

    path = r"C:\Users\chris\Desktop\ECEC487Materials\RecognitionProject\songs\demosongs\\" + "new" + os.path.basename(title)
    librosa.output.write_wav(path,y,sr)

    stft = np.abs(librosa.stft(y, n_fft=2048, center=False))
    librosa.display.specshow(librosa.amplitude_to_db(stft, ref=np.max), y_axis='log', x_axis='time')

    print(stft.shape)
