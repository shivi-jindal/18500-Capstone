import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fftpack import fft

# List of file paths
filepaths = ["clear1.wav"]
colors = ['b']  # Different colors for each note

plt.figure(figsize=(10, 5))  # Create a single figure

for file_path, color in zip(filepaths, colors):
    # Load audio file
    sample_rate, data = wavfile.read(file_path)

    # Convert to mono if stereo
    if len(data.shape) > 1:
        data = data.mean(axis=1)

    # Perform FFT
    N = len(data)
    freqs = np.fft.fftfreq(N, 1/sample_rate)
    fft_values = np.abs(fft(data))

    # Keep only the positive half of the spectrum
    positive_freqs = freqs[:N // 2]
    positive_fft_values = fft_values[:N // 2]

    # Find fundamental frequency
    fundamental_index = np.argmax(positive_fft_values)
    fundamental_freq = positive_freqs[fundamental_index]

    # Identify harmonics
    harmonics = [fundamental_freq * i for i in range(1, 10)]  # First 10 harmonics
    harmonic_amplitudes = [positive_fft_values[np.argmin(np.abs(positive_freqs - h))] for h in harmonics]

    # Plot FFT Spectrum
    plt.plot(positive_freqs, positive_fft_values, label=f"{file_path.split('.')[0]}", color=color, alpha=0.6)

    # Mark Harmonics
    plt.scatter(harmonics, harmonic_amplitudes, color=color, edgecolors="k", label=f"{file_path.split('.')[0]} Harmonics")

# Final plot adjustments
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.title("Harmonics of Flute Notes")
plt.legend()
plt.xlim(0, 2000)  # Adjust as needed
plt.grid()
plt.show()
