import librosa
import librosa.display
import numpy as np
import scipy.signal as signal
import noisereduce as nr
import soundfile as sf
import matplotlib.pyplot as plt
from pydub import AudioSegment


class Denoising:
    def __init__(self):
        return

    def convert_m4a_to_wav(self, input_file, new_filepath):
        audio = AudioSegment.from_file(input_file, format="m4a")
        audio.export(new_filepath, format="wav")

    def load_audio(self, file_path):
        y, sr = librosa.load(file_path, sr=None)
        return y, sr

    def adaptive_noise_reduction(self, y, sr):
        noise_part = y[:sr]

        y_denoised = nr.reduce_noise(y=y, sr=sr, y_noise=noise_part, prop_decrease=0.5)
        return y_denoised

    def bandpass_filter(self, y, sr, lowcut=260, highcut=2100, order=6):
        nyquist = 0.5 * sr  
        low = lowcut / nyquist
        high = highcut / nyquist

        b, a = signal.butter(order, [low, high], btype='band')  
        y_filtered = signal.lfilter(b, a, y)  

        return y_filtered

    def spectral_gate(self, y, sr, reduction_factor=0.2):
        
        S = librosa.stft(y)
        magnitude, phase = librosa.magphase(S)


        noise_sample = y[:5*sr]  
        noise_stft = np.abs(librosa.stft(noise_sample)).mean(axis=1)

        magnitude_denoised = np.maximum(magnitude - reduction_factor * noise_stft[:, None], 0)

        return librosa.istft(magnitude_denoised * phase)


    def noise_suppression_pipeline(self, input_file):
        if input_file.endswith(".m4a"):
            wav_file = input_file.replace(".m4a", ".wav")
            self.convert_m4a_to_wav(input_file, wav_file)
            input_file = wav_file
        
        y, sr = self.load_audio(input_file)
        
        # y_filtered = self.bandpass_filter(y, sr)
        return y, sr
