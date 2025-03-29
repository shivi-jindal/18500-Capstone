import librosa
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class Pitch: 
    def compute_stft(self, signal, sr, hop_size=512, win_size=1024):
        D = librosa.stft(signal, n_fft=win_size, hop_length=hop_size, win_length=win_size)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=win_size)
        return D, freqs

    def apply_comb_filter(self, freqs, spectrum, fundamental_range=(50, 1000)):
        """Find the fundamental frequency using a comb filter."""
        best_freq = None
        max_response = -np.inf

        for f0 in np.arange(fundamental_range[0], fundamental_range[1], 1):
            comb_filter = np.sum([spectrum[np.argmin(np.abs(freqs - (k * f0)))] for k in range(1, 6)])
            
            if comb_filter > max_response:
                max_response = comb_filter
                best_freq = f0

        return best_freq

    def detect_notes(self, signal, sr, seg_times, hop_size=512, win_size=1024):
        note_frequencies = []

        for i in range(len(seg_times) - 1):
            start_sample = int((seg_times[i] / 1000) * sr)
            end_sample = int((seg_times[i + 1] / 1000) * sr)
            
            segment = signal[start_sample:end_sample]
            
            D, freqs = self.compute_stft(segment, sr, hop_size, win_size)
            spectrum = np.mean(np.abs(D), axis=1)
            
            fundamental_freq = self.apply_comb_filter(freqs, spectrum)

            note_frequencies.append(fundamental_freq)
        
        return note_frequencies

    def freq_to_note(self, freq):
        A4 = 440.0 
        if freq is None or freq <= 0:
            return "Unknown"
        
        note_num = round(12 * np.log2(freq / A4)) + 69 
        return note_num

    def plot_comb_filter_response(self, freqs, spectrum, fundamental_range=(50, 1000)):
        
        response_values = []
        candidate_freqs = np.arange(fundamental_range[0], fundamental_range[1], 1)

        for f0 in candidate_freqs:
            comb_response = np.sum([spectrum[np.argmin(np.abs(freqs - (k * f0)))] for k in range(1, 6)])
            response_values.append(comb_response)

        plt.figure(figsize=(10, 5))
        plt.plot(candidate_freqs, response_values, label="Comb Filter Response", color="b")
        plt.xlabel("Fundamental Frequency (Hz)")
        plt.ylabel("Filter Response Strength")
        plt.title("Comb Filter Response for Pitch Estimation")
        plt.grid(True)
        plt.legend()
        plt.show()


# rms_vals, sr, og_signal = perform_rms("../Audio/Songs/twinkle.m4a")
# segs = calculate_new_notes(rms_vals, 512, sr)
# detected_frequencies = detect_notes(og_signal, sr, segs)
# detected_notes = [freq_to_note(f) for f in detected_frequencies] #list of tuples of (note_num, note)
# print("Detected Notes:", detected_notes)