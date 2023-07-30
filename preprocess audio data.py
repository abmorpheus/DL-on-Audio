import matplotlib.pyplot as plt
import librosa, librosa.display
import numpy as np

file = 'D:\COLLEGE\PROJECTS\Audio DL Basics\sound.wav'

# waveform
signal, sr = librosa.load(file, sr = 22050) # sr = sample rate | sinal = sr * seconds
# librosa.display.waveshow(signal, sr = sr)
# plt.xlabel('Time')
# plt.ylabel('Amplitude')
# plt.show()

# fast fourier transform -> spectrogram
fft = np.fft.fft(signal)

magnitude = np.abs(fft) # indicates contribution of each frequency
frequency = np.linspace(0, sr, len(magnitude))
left_freq = frequency[:int(len(frequency)/2)]
left_magnitude = magnitude[:int(len(magnitude)/2)]
# plt.plot(left_freq, left_magnitude)
# plt.xlabel('Frequency')
# plt.ylabel('Magnitude')
# plt.show()

# stft = short time fourier transform
# stft -> spectrogram
n_fft = 2048 # number of samples considering while performing a single fourier transform
hop_length = 512 # how much we are sliding to the left
stft = librosa.core.stft(signal, hop_length = hop_length, n_fft = n_fft)
sprectrogram = np.abs(stft)

log_spectrogram = librosa.amplitude_to_db(sprectrogram)

# librosa.display.specshow(log_spectrogram, sr = sr, hop_length = hop_length)
# plt.xlabel('Time')
# plt.ylabel('Frequency')
# plt.colorbar()
# plt.show()

# MFCCs = Mel-Frequency Cepstral Coefficients
MFCCs = librosa.feature.mfcc(y = signal, n_fft = n_fft, hop_length = hop_length, n_mfcc = 14)

# librosa.display.specshow(MFCCs, sr = sr, hop_length = hop_length)
# plt.xlabel('Time')
# plt.ylabel('MFCC')
# plt.colorbar()
# plt.show()





