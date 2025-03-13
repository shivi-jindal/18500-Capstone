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
    predicted_starts = []
    
    for i in range(2, len(rms_vals)):
        # print(rms_vals[i-1])
        if rms_vals[i] - rms_vals[i - 1] > 0.07:
            predicted_starts.append(i * hop_size/sr)
    return predicted_starts


def plot_rms(rms_values, sr, hop_size):
    ''' graphing for just the rms values '''
    # time = np.arange(0, len(rms_values) * hop_size, hop_size) / sr

    # plt.figure(figsize=(10, 6))
    # plt.plot(time, rms_values, label='RMS')
    # plt.title('RMS of Audio Signal')
    # plt.xlabel('Time (seconds)')
    # plt.ylabel('RMS')
    # plt.grid(True)
    # plt.legend()
    # plt.show()

    ''' graphing for just the rms values and peak identification test 1 '''
    # time = np.arange(0, len(rms_values) * hop_size, hop_size) / sr

    # threshold = 0.07
    # min_time_diff = 0.05  # 5ms = 0.005 seconds

    # spike_times = []
    # for i in range(len(rms_values) - 1):
    #     current_time = time[i]
    #     for j in range(i + 1, len(rms_values)):
    #         time_diff = time[j] - current_time
    #         if time_diff >= min_time_diff:
    #             rms_diff = rms_values[j] - rms_values[i]
    #             if rms_diff > threshold:
    #                 spike_times.append(time[j])
    #             break  # Only need the first RMS value that meets the time difference condition
    # for spike in spike_times:
    #     print(f"Spike at time {spike:.3f} seconds")

    # plt.figure(figsize=(10, 6))
    # plt.plot(time, rms_values, label='RMS')

    # plt.scatter(spike_times, [rms_values[np.where(time == spike)[0][0]] for spike in spike_times], color='red', label='Spikes', zorder=5)

    # plt.title('RMS of Audio Signal')
    # plt.xlabel('Time (seconds)')
    # plt.ylabel('RMS')
    # plt.grid(True)
    # plt.legend()
    # plt.show()

    '''detecing note changes by looking for when the rms is virtually zero, but picks up moments of silence'''

    time_seconds = np.arange(0, len(rms_values) * hop_size, hop_size) / sr

    # Convert the time to milliseconds for plotting
    time_ms = time_seconds * 1000

    # Define an epsilon for identifying near-zero RMS values
    epsilon = 0.01  # Tolerance for RMS values close to zero

    # Initialize a list to store spike times (when RMS is near zero)
    spike_times = []

    # Iterate through the RMS values to identify near-zero spikes
    for i in range(len(rms_values)):
        if rms_values[i] < epsilon:  # Check if RMS is close to zero
            if (len(spike_times) == 0 or (time_ms[i] - spike_times[-1]) >= 100):
                spike_times.append(time_ms[i])

    # Print out the times of the near-zero RMS spikes
    print(f"Times where RMS is near zero (within epsilon = {epsilon}):")
    for spike in spike_times:
        print(f"Spike at time {spike:.3f} milliseconds")

    plt.figure(figsize=(10, 6))
    plt.plot(time_ms, rms_values, label='RMS')

    # Mark the steep RMS increase after silence with vertical lines
    for spike_time in spike_times:
        # Find the index of the spike time in the time array
        spike_index = np.where(time_ms == spike_time)[0][0]
        
        # Plot a vertical line at the spike time
        plt.axvline(x=spike_time, color='red', linestyle='-', lw=2)

    plt.title('RMS of Audio Signal with Steep Increases After Silence (Vertical Lines)')
    plt.xlabel('Time (milliseconds)')
    plt.ylabel('RMS')
    plt.grid(True)
    plt.legend()
    plt.show()
   




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

rms_vals, sr, og_signal = perform_rms("Segmentation.m4a")
plot_rms(rms_vals, sr, 512)
# plot_rms_and_regular(og_signal, rms_vals, sr, 512)
segs = calculate_new_notes(rms_vals, 512, sr)
# print(segs)
    
    