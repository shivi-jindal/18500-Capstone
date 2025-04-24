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


class Segmentation:
    def __init__(self):
        return

    def perform_rms(self, signal, sr):
        # maybe change these
        window_size = 1024
        hop_size = 512
        rms_values = []
        for start in range(0, len(signal) - window_size, hop_size):
            window = signal[start:start + window_size]
            rms = np.sqrt(np.mean(window**2))
            rms_values.append(rms)
        return np.array(rms_values), sr, signal
    
    def perform_ste(self, signal, sr):
        window_size = 1024
        hop_size = 512
        ste_values = []

        for start in range(0, len(signal) - window_size, hop_size):
            window = signal[start:start + window_size]
            ste = np.sum(window**2)
            ste_values.append(ste)

        return np.array(ste_values), sr, signal

    def calculate_new_notes_rms(self, rms_vals, hop_size, sr):
        time_seconds = np.arange(0, len(rms_vals) * hop_size, hop_size) / sr
        time_ms = time_seconds * 1000
        epsilon = 0.03 
        valid_spikes = []
        # difference from the near zero value to the peak of the rms siganl
        peak_difference_threshold = 0.06 # changing value based on bpm?
        # not counting the beginning of a note time 
        min_spike_difference = 500
        # making sure the nearest peak is closer
        max_time_difference = 150
        for i in range(len(rms_vals)):
            if rms_vals[i] < epsilon:  # Check if RMS is close to zero
                # check if it is significantly different than last time
                if (len(valid_spikes) == 0 or (time_ms[i] - valid_spikes[-1]) >= min_spike_difference):
                    for j in range(i + 1, len(rms_vals)):
                        # find the nearest peak from the spike
                        if j < len(rms_vals) - 1 and j >= 1 and rms_vals[j] > rms_vals[j - 1] and rms_vals[j] > rms_vals[j + 1]:  # Local peak
                            peak_value = rms_vals[j]
                            spike_value = rms_vals[i]
                            # check if this was a significant increase and if spike wasn't super far away (within 150 ms)
                            if peak_value - spike_value > peak_difference_threshold: #and abs(time_ms[j] - time_ms[i]) <= max_time_difference:
                                valid_spikes.append(time_ms[i]) #adding in the beginning of zero time, but make add in peak time/average of the two?
                                break
        return valid_spikes
    
   

    def plot_rms(self, rms_values, sr, hop_size):
        '''detecing note changes by looking for when the rms is virtually zero, but picks up moments of silence'''

        time_seconds = np.arange(0, len(rms_values) * hop_size, hop_size) / sr

        # Convert the time to milliseconds for plotting
        time_ms = time_seconds * 1000

        # Define an epsilon for identifying near-zero RMS values
        epsilon = 0.03
        valid_spikes = []
        peak_difference_threshold = 0.06
        min_spike_difference = 500
        max_time_difference = 100
        
        # Iterate through the RMS values to identify near-zero spikes
        for i in range(len(rms_values)):
            if rms_values[i] < epsilon:  # Check if RMS is close to zero
                # check if it is significantly different than last time
                if (len(valid_spikes) == 0) or (time_ms[i] - valid_spikes[-1]) >= min_spike_difference:
                    # spike_times.append(time_ms[i])
                    for j in range(i + 1, len(rms_values)):
                        # find the nearest peak from the spike
                        if j < len(rms_values) - 1 and j >= 1 and rms_values[j] > rms_values[j - 1] and rms_values[j] > rms_values[j + 1]:  # Local peak
                            peak_value = rms_values[j]
                            spike_value = rms_values[i]
                            # check if this was a significant increase and if spike wasn't super far away (within 150 ms)
                            if peak_value - spike_value > peak_difference_threshold: #and abs(time_ms[j] - time_ms[i]) <= max_time_difference:
                                valid_spikes.append(time_ms[i]) #adding in the beginning of zero time, but make add in peak time/average of the two?
                            break

        # Print out the times of the near-zero RMS spikes
        # print(f"Times where RMS is near zero (within epsilon = {epsilon}):")
        # for spike in valid_spikes:
        #     print(f"Spike at time {spike/1000:.3f} milliseconds")

        plt.figure(figsize=(10, 6))
        plt.plot(time_ms, rms_values, label='RMS')


        # Mark the steep RMS increase after silence with vertical lines
        for spike_time in valid_spikes:
            # Find the index of the spike time in the time array
            spike_index = np.where(time_ms == spike_time)[0][0]
            
            # Plot a vertical line at the spike time
            plt.axvline(x=spike_time, color='red', linestyle='-', lw=2)

        plt.title('RMS of Audio Signal')
        plt.xlabel('Time (milliseconds)')
        plt.ylabel('RMS')
        plt.grid(True)
        plt.legend()
        plt.show()
        return valid_spikes

    def plot_rms_and_regular(self, audio_signal, rms_values, sr, hop_size):
        time_audio = np.arange(0, len(audio_signal)) / sr
        time_rms = np.arange(0, len(rms_values) * hop_size, hop_size) / sr

        plt.figure(figsize=(15, 6))

        plt.subplot(1, 2, 1)
        plt.plot(time_audio, audio_signal, label='Audio Signal')
        plt.title('Audio Signal')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(time_rms, rms_values, label='RMS', color='r')
        plt.title('RMS of Audio Signal')
        plt.xlabel('Time (seconds)')
        plt.ylabel('RMS')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()
    
    def calculate_new_notes_ste(self, ste_vals, y, sr, hop_size, bpm):
        time_seconds = np.arange(0, len(ste_vals) * hop_size, hop_size) / sr
        time_ms = time_seconds * 1000
       
        epsilon = 0.5 
        valid_spikes = []
        # not counting the beginning of a note time 
        # min_spike_difference = 150 # need to change this based on BPM, 100 too small for long notes but good for fast tempos
        beat_duration_ms = 60000 / bpm  # duration of a beat in ms
        min_spike_difference = beat_duration_ms * 1/2  # adjust this multiplier to fine-tune

        max_time_difference = 150
        for i in range(len(ste_vals)):
            if ste_vals[i] < epsilon:  # Check if RMS is close to zero
                if (len(valid_spikes) == 0) or (time_ms[i] - valid_spikes[-1]) >= min_spike_difference:
                    for j in range(i + 1, len(ste_vals)):
                        # find the nearest peak from the spike
                        if (len(valid_spikes) == 0 or abs(time_ms[i] - valid_spikes[-1]) >= min_spike_difference):
                            if j >= 1 and ste_vals[j] > ste_vals[j - 1] and ste_vals[j] > epsilon:  # check for min height
                                # check if this was a significant increase and if spike wasn't super far away (within 150 ms)
                                # if peak_value - spike_value > peak_difference_threshold and abs(time_ms[j] - time_ms[i]) <= max_time_difference:
                                if abs(time_ms[j] - time_ms[i]) <= max_time_difference:  
                                    valid_spikes.append(time_ms[i]) #adding in the beginning of zero time, but make add in peak time/average of the two?
                                    break
        valid_spikes.append(time_ms[i - 1])
        slurred_notes = []
        # look for local minimums (could be slurred notes)
        for m in range(len(valid_spikes) - 1):
            start_sample = int((valid_spikes[m] / 1000) * sr/hop_size)
            end_sample = int((valid_spikes[m + 1] / 1000) * sr/hop_size)
            segment = ste_vals[start_sample:end_sample]
            for t in range(1, len(segment) - 1):
                if segment[t] < segment[t - 1] and segment[t] < segment[t + 1] and segment[t] > epsilon and segment[t + 1] > epsilon:
                    slurred_notes += [(start_sample, end_sample)]
                    break
        
        # check the end as well - might be excessive
        if len(valid_spikes) >= 2:
            start_sample = int((valid_spikes[-2] / 1000) * sr/hop_size)
            end_sample = int((valid_spikes[-1] / 1000) * sr/hop_size)
            segment = ste_vals[start_sample:end_sample]
            for t in range(1, len(segment) - 1):
                if segment[t] < segment[t - 1] and segment[t] < segment[t + 1] and segment[t] > epsilon and segment[t + 1] > epsilon:
                    slurred_notes += [(start_sample, end_sample)]
                    break
        freq_changes = self.detect_pitch_changes(slurred_notes, y, sr, 512, min_spike_difference)
        # print(freq_changes)
        valid_spikes += freq_changes
        valid_spikes = sorted(valid_spikes)
        # graphing code

        segments = []
        for i in range(1, len(valid_spikes)):
            if abs(valid_spikes[i] - valid_spikes[i - 1]) >= 100:
                segments += [valid_spikes[i-1]]

        # plt.figure(figsize=(10, 6))
        # plt.plot(time_ms, ste_vals, label='STE')

        # for spike_time in segments:
            
        #     # Plot a vertical line at the spike time
        #     plt.axvline(x=spike_time, color='red', linestyle='-', lw=2)
        

        # plt.title('STE of Audio Signal')
        # plt.xlabel('Time (milliseconds)')
        # plt.ylabel('STE')
        # plt.grid(True)
        # plt.show()                   
        return segments

    def detect_pitch_changes(self, note_times, y, sr, hop_size, min_diff):
        win_size = 4056
        D = librosa.stft(y, n_fft=win_size, hop_length=hop_size, win_length=win_size)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=win_size)
        D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        time_ms = librosa.frames_to_time(np.arange(D_db.shape[-1]), sr=sr, hop_length=hop_size) * 1000  # in milliseconds
        freq_changes = []
        threshold = 1
        # print(min_diff)
        if min_diff > 300:
            threshold += 0.1
        prev_freq = None
        for start_time, end_time in note_times:
            prev_freq = None
            for t in range(start_time, end_time + 1):
                freq_index = np.argmax(D_db[:, t])
                current_freq = freqs[freq_index]
                if prev_freq is not None:
                    max_freq = abs(max(current_freq, prev_freq))
                    min_freq = abs(min(current_freq, prev_freq))
                    if min_freq == 0:
                        min_freq = 0.1
                if prev_freq is not None and max_freq/min_freq > threshold and (len(freq_changes) == 0 or abs(time_ms[t] - freq_changes[-1]) >= min_diff):  
                    # Save the time of change if it exceeds the threshold
                    freq_changes.append(time_ms[t])

                # Update the previous frequency for the next comparison
                prev_freq = current_freq
        # print(freq_changes)
        return freq_changes
            
    def segment_notes(self, signal, sr, bpm):
        ste_vals, sr, og_signal = self.perform_ste(signal, sr)
        segs = self.calculate_new_notes_ste(ste_vals, signal, sr, 512, bpm)
        # segs += [len(og_signal) * 512 * 1000 / sr] # multiply by the hop_size and convert to ms
        return ste_vals, sr, og_signal, segs



# segmentation = Segmentation()
# y, sr = load_audio("../Audio/Songs/Slurred_Scale.wav")
# ste_vals, sr, og_signal = segmentation.perform_ste(y, sr)
# segmentation.calculate_new_notes_ste(ste_vals, y, sr, 512, bpm = 100)
 
#try modifying ratio ... or smoothing of the signal