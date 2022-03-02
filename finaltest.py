import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

y, sr = librosa.load(r"C:\Users\chris\Desktop\ECEC487Materials\RecognitionProject\songs\MachineMixes\AroundTheWorldxBacktoBlack.wav")

stft = np.abs(librosa.stft(y, n_fft=2048, center=False))
librosa.display.specshow(librosa.amplitude_to_db(stft, ref=np.max), y_axis='log', x_axis='time')
plt.title('STFT - Around the World x Back to Black')
plt.tight_layout()
plt.show()
