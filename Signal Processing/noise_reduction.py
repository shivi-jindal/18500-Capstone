import librosa
import librosa.display
import numpy as np
import scipy.signal as signal
import noisereduce as nr
import soundfile as sf
import matplotlib.pyplot as plt
from pydub import AudioSegment

def convert_m4a_to_wav(input_file, output_file):
    audio = AudioSegment.from_file(input_file, format="m4a")
    audio.export(output_file, format="wav")


def load_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    return y, sr


def adaptive_noise_reduction(y, sr):
    # print(len(y))
    noise_part = y[:sr]
    # print(f"Noise segment shape: {noise_part.shape}")

    y_denoised = nr.reduce_noise(y=y, sr=sr, y_noise=noise_part, prop_decrease=0.5)
    return y_denoised

def bandpass_filter(y, sr, lowcut=260, highcut=2100, order=6):
    nyquist = 0.5 * sr  
    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = signal.butter(order, [low, high], btype='band')  
    y_filtered = signal.lfilter(b, a, y)  

    return y_filtered

def spectral_gate(y, sr, reduction_factor=0.2):
    
    S = librosa.stft(y)
    magnitude, phase = librosa.magphase(S)


    noise_sample = y[:5*sr]  
    noise_stft = np.abs(librosa.stft(noise_sample)).mean(axis=1)

    magnitude_denoised = np.maximum(magnitude - reduction_factor * noise_stft[:, None], 0)

    return librosa.istft(magnitude_denoised * phase)


def noise_suppression_pipeline(input_file, output_file):
    if input_file.endswith(".m4a"):
        wav_file = input_file.replace(".m4a", ".wav")
        convert_m4a_to_wav(input_file, wav_file)
        input_file = wav_file
    
    y, sr = load_audio(input_file)
    
    y_filtered = bandpass_filter(y, sr)

    y_final = spectral_gate(y_filtered, sr)

    y_final = adaptive_noise_reduction(y_final, sr)

    sf.write(output_file, y_final, sr)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(2, 1, 1)
    plt.plot(y, label="Original", alpha=0.6)
    plt.title("Original Audio Waveform")
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(y_final, label="Denoised", alpha=0.8, color='g')
    plt.title("Denoised Audio Waveform")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

input_file = "../Audio/fluteD_noisy.m4a" 
output_file = "flute_denoised3.wav"
noise_suppression_pipeline(input_file, output_file)
