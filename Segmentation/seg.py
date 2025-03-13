import librosa
import librosa.display
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from pydub import AudioSegment

def convert_m4a_to_wav(input_file, output_file):
    audio = AudioSegment.from_file(input_file, format="m4a")
    audio.export(output_file, format="wav")


def load_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    return y, sr

def perform_rms(input_file):
    if input_file.endswith(".m4a"):
        wav_file = input_file.replace(".m4a", ".wav")
        convert_m4a_to_wav(input_file, wav_file)
        input_file = wav_file
    signal, sr = load_audio(input_file)
    # maybe change these
    window_size = 1024
    hop_size = 512
    rms_values = []
    for start in range(0, len(signal) - window_size, hop_size):
        window = signal[start:start + window_size]
        rms = np.sqrt(np.mean(window**2))
        rms_values.append(rms)
    return np.array(rms_values), sr

def plot_rms(rms_values, sr, hop_size):
    """Plot the RMS values."""
    # Time vector for plotting
    time = np.arange(0, len(rms_values) * hop_size, hop_size) / sr

    plt.figure(figsize=(10, 6))
    plt.plot(time, rms_values, label='RMS')
    plt.title('RMS of Audio Signal')
    plt.xlabel('Time (seconds)')
    plt.ylabel('RMS')
    plt.grid(True)
    plt.legend()
    plt.show()

rms_vals, sr = perform_rms("Segmentation.m4a")
plot_rms(rms_vals, sr, 512)
    
    