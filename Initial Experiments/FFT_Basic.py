import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fftpack import fft

filepaths = ["../Audio/Recordings/bflat.wav", "../Audio/Recordings/c.wav", "../Audio/Recordings/d.wav", "../Audio/Recordings/eflat.wav", "../Audio/Recordings/f.wav", "../Audio/Recordings/g.wav"]
colors = ['b', 'g', 'r', 'c', 'm', 'y']

plt.figure(figsize=(10, 5)) 

for file_path, color in zip(filepaths, colors):
    
    sample_rate, data = wavfile.read(file_path)

    if len(data.shape) > 1:
        data = data.mean(axis=1)

    # FFT
    N = len(data)
    freqs = np.fft.fftfreq(N, 1/sample_rate)
    fft_values = np.abs(fft(data))

    # Keep only the positive half of the spectrum
    positive_freqs = freqs[:N // 2]
    positive_fft_values = fft_values[:N // 2]

    fundamental_index = np.argmax(positive_fft_values)
    fundamental_freq = positive_freqs[fundamental_index]

    harmonics = [fundamental_freq * i for i in range(1, 10)]  # First 10 harmonics
    harmonic_amplitudes = [positive_fft_values[np.argmin(np.abs(positive_freqs - h))] for h in harmonics]

    plt.plot(positive_freqs, positive_fft_values, label=f"{file_path.split('.')[0]}", color=color, alpha=0.6)

    plt.scatter(harmonics, harmonic_amplitudes, color=color, edgecolors="k", label=f"{file_path.split('.')[0]} Harmonics")

plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.title("Harmonics of Flute Notes")
plt.legend()
plt.xlim(0, 2000)  
plt.grid()
plt.show()
