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
    return np.array(rms_values), sr, signal

def calculate_new_notes(rms_vals, hop_size, sr):
    time_seconds = np.arange(0, len(rms_vals) * hop_size, hop_size) / sr
    time_ms = time_seconds * 1000
    epsilon = 0.03  # Tolerance for RMS values close to zero

    # Initialize a list to store spike times (when RMS is near zero)
    
    valid_spikes = []
    # difference from the near zero value to the peak of the rms siganl
    peak_difference_threshold = 0.06 # changing value based on bpm?
    # not counting the beginning of a note time 
    min_spike_difference = 500
    # making sure the nearest peak is closer
    max_time_difference = 150
    # Iterate through the RMS values to identify near-zero spikes
    for i in range(len(rms_vals)):
        if rms_vals[i] < epsilon:  # Check if RMS is close to zero
            # check if it is significantly different than last time
            if (len(valid_spikes) == 0 or (time_ms[i] - valid_spikes[-1]) >= min_spike_difference):
                # spike_times.append(time_ms[i])
                for j in range(i + 1, len(rms_vals)):
                    # find the nearest peak from the spike
                    if rms_vals[j] > rms_vals[j - 1] and rms_vals[j] > rms_vals[j + 1]:  # Local peak
                        peak_value = rms_vals[j]
                        spike_value = rms_vals[i]
                        # check if this was a significant increase and if spike wasn't super far away (within 150 ms)
                        if peak_value - spike_value > peak_difference_threshold and abs(time_ms[j] - time_ms[i]) <= max_time_difference:
                            valid_spikes.append(time_ms[i]) #adding in the beginning of zero time, but make add in peak time/average of the two?
                            break
    return valid_spikes

def plot_rms(rms_values, sr, hop_size):
    '''detecing note changes by looking for when the rms is virtually zero, but picks up moments of silence'''

    time_seconds = np.arange(0, len(rms_values) * hop_size, hop_size) / sr

    # Convert the time to milliseconds for plotting
    time_ms = time_seconds * 1000

    # Define an epsilon for identifying near-zero RMS values
    epsilon = 0.04  # Tolerance for RMS values close to zero

    # Initialize a list to store spike times (when RMS is near zero)
    
    valid_spikes = []
    peak_difference_threshold = 0.03
    min_spike_difference = 500
    max_time_difference = 100
    
    # Iterate through the RMS values to identify near-zero spikes
    for i in range(len(rms_values)):
        if rms_values[i] < epsilon:  # Check if RMS is close to zero
            # check if it is significantly different than last time
            if (len(valid_spikes) == 0 or (time_ms[i] - valid_spikes[-1]) >= min_spike_difference):
                # spike_times.append(time_ms[i])
                for j in range(i + 1, len(rms_values)):
                    # find the nearest peak from the spike
                    if rms_values[j] > rms_values[j - 1] and rms_values[j] > rms_values[j + 1]:  # Local peak
                        peak_value = rms_values[j]
                        spike_value = rms_values[i]
                        # check if this was a significant increase and if spike wasn't super far away (within 150 ms)
                        if peak_value - spike_value > peak_difference_threshold and abs(time_ms[j] - time_ms[i]) <= max_time_difference:
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

def plot_rms_and_regular(audio_signal, rms_values, sr, hop_size):
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

rms_vals, sr, og_signal = perform_rms("../Audio/Songs/hotcross_interface.m4a")
plot_rms(rms_vals, sr, 512)
# plot_rms_and_regular(og_signal, rms_vals, sr, 512)
#segs = calculate_new_notes(rms_vals, 512, sr)
    