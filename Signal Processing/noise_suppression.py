import librosa
import librosa.display
import numpy as np
import scipy.signal as signal
import noisereduce as nr
import soundfile as sf
import matplotlib.pyplot as plt
from pydub import AudioSegment

# .m4a conversion to .wav
def convert_m4a_to_wav(input_file, output_file):
    audio = AudioSegment.from_file(input_file, format="m4a")
    audio.export(output_file, format="wav")

#load da audio file
def load_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    return y, sr

#spectral subtraction
def spectral_subtraction(y, sr):
    noise_part = y[:sr]  # Assume the first second is noise-only
    noise_spec = np.abs(librosa.stft(noise_part)).mean(axis=1)
    
    S = librosa.stft(y)
    S_denoised = np.maximum(np.abs(S) - noise_spec[:, None], 0) * np.exp(1j * np.angle(S))
    
    return librosa.istft(S_denoised)

#wiener filtering
def wiener_filter(y, sr):
    return signal.wiener(y)

#apply noise suppression and plot waveforms
def noise_suppression_pipeline(input_file, output_file):
    # Convert if input is .m4a
    if input_file.endswith(".m4a"):
        wav_file = input_file.replace(".m4a", ".wav")
        convert_m4a_to_wav(input_file, wav_file)
        input_file = wav_file
    
    y, sr = load_audio(input_file)
    
    y_denoised = spectral_subtraction(y, sr)
    
    y_final = wiener_filter(y_denoised, sr)
    
    # Save output
    sf.write(output_file, y_final, sr)
    print(f"Denoised audio saved to {output_file}")
    
    # Plot waveforms
    plt.figure(figsize=(10, 4))
    plt.plot(y, label="Original", alpha=0.6)
    plt.plot(y_final, label="Denoised", alpha=0.8)
    plt.legend()
    plt.show()

input_file = "flute_noisy.m4a"
output_file = "flute_denoised.wav"
