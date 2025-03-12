import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Read the WAV file
sample_rate, audio_data = wavfile.read('clear1.wav')

# Check if audio data is stereo (2 channels) or mono (1 channel)
if len(audio_data.shape) == 2:
    # If stereo, we take one channel (e.g., left channel)
    audio_data = audio_data[:, 0]

# Perform FFT
n = len(audio_data)
frequencies = np.fft.fftfreq(n, d=1/sample_rate)
fft_values = np.fft.fft(audio_data)

# Create a mask for 600 Hz and 1200 Hz
def filter_frequency(frequency_range):
    # Create a mask that isolates the specified frequency range
    mask = (frequencies >= frequency_range[0]) & (frequencies <= frequency_range[1])
    return mask

# Frequencies for 600 Hz and 1200 Hz ranges
mask_600Hz = filter_frequency((590, 610))  # A small range around 600 Hz
mask_1200Hz = filter_frequency((1190, 1210))  # A small range around 1200 Hz

# Create filtered FFT values for both 600 Hz and 1200 Hz
filtered_fft_600Hz = fft_values.copy()
filtered_fft_600Hz[~mask_600Hz] = 0  # Zero out all other frequencies

filtered_fft_1200Hz = fft_values.copy()
filtered_fft_1200Hz[~mask_1200Hz] = 0  # Zero out all other frequencies

# Perform IFFT to get the time-domain signal for each frequency component
filtered_audio_600Hz = np.fft.ifft(filtered_fft_600Hz).real
filtered_audio_1200Hz = np.fft.ifft(filtered_fft_1200Hz).real

# Generate time axis in seconds
time = np.linspace(0, len(audio_data) / sample_rate, num=len(audio_data))

# Plot the two graphs
plt.figure(figsize=(12, 8))

# Plot for 600 Hz
plt.subplot(2, 1, 1)
plt.plot(time, filtered_audio_600Hz)
plt.title('Signal at 600 Hz')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)

# Plot for 1200 Hz
plt.subplot(2, 1, 2)
plt.plot(time, filtered_audio_1200Hz)
plt.title('Signal at 1200 Hz')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)

plt.tight_layout()
plt.show()
