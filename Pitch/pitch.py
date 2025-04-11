import librosa
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import copy
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
top_freq = 0
class Pitch: 
    def compute_stft(self, signal, sr, hop_size=512, win_size=1024):
        D = librosa.stft(signal, n_fft=win_size, hop_length=hop_size, win_length=win_size)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=win_size)
        return D, freqs

    def apply_comb_filter(self, freqs, spectrum, fundamental_range=(50, 1000)):
        best_freq = None
        max_response = -np.inf

        for f0 in np.arange(fundamental_range[0], fundamental_range[1], 1):
            comb_filter = 0
            for k in range(1,6):
                diffs = np.abs(freqs - k*f0) #compute abs diffs between actual freq and freq bins
                min_fundamental = np.argmin(diffs) #find the closest frequency bin (k) to the expected harmonic freq 
                comb_filter += spectrum[min_fundamental] #sum up the freq response
            
            if comb_filter > max_response: #the f0 that gives highest sum is the best fund freq
                max_response = comb_filter
                best_freq = f0
        filtered_spectrum = copy.deepcopy(spectrum)
        for k in range(2, 6):  # Removing harmonics (2nd to 5th)
            harmonic_freq = k * best_freq
            diffs = np.abs(freqs - harmonic_freq)
            min_harmonic = np.argmin(diffs)
            filtered_spectrum[min_harmonic] = 0  # Suppress harmonic frequencies

        return best_freq, spectrum, filtered_spectrum
    
    def plot_comb_filter(self, freqs, spectrum, filtered_spectrum, fundamental_freq):
        # Convert the spectrum to decibels
        spectrum_db = librosa.amplitude_to_db(spectrum, ref=np.max)

        # Plot the decibels vs frequency for the filtered segment
        plt.figure(figsize=(10, 6))
        plt.plot(freqs, spectrum_db, label="Original Spectrum")
        plt.title("Original Spectrum - Decibels vs Frequency")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (dB)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        filtered_db = librosa.amplitude_to_db(filtered_spectrum, ref=np.max)

        # Plot the decibels vs frequency for the filtered segment
        plt.figure(figsize=(10, 6))
        plt.plot(freqs, filtered_db, label="Filtered Spectrum")
        plt.axvline(fundamental_freq, color='r', linestyle='--', label=f"Fundamental: {fundamental_freq:.2f} Hz")
        plt.title("Filtered Spectrum - Decibels vs Frequency")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (dB)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def detect_notes(self, signal, sr, seg_times, hop_size=512, win_size=1024):
        note_frequencies = []
        
        # Iterate through each segment (note)
        for i in range(len(seg_times) - 1):
            start_sample = int((seg_times[i] / 1000) * sr)
            end_sample = int((seg_times[i + 1] / 1000) * sr)
            
            segment = signal[start_sample:end_sample]  # Extract the segment based on start/end time
            
            # Compute STFT on the segment
            D, freqs = self.compute_stft(segment, sr, hop_size, win_size)
            spectrum = np.mean(np.abs(D), axis=1)  # Compute avg magnitude at each frequency bin
            
            fundamental_freq, spectrum, filtered_spectrum = self.apply_comb_filter(freqs, spectrum)
            # self.plot_comb_filter(freqs, spectrum, filtered_spectrum, fundamental_freq)

            note_frequencies.append(fundamental_freq)

        return note_frequencies

    def freq_to_note(self, freq):
        A4 = 440.0 
        if freq is None or freq <= 0:
            return "Unknown"
        
        note_num = round(12 * np.log2(freq / A4)) + 69 
        return note_num