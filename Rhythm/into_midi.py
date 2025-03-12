import numpy as np
from scipy.io import wavfile
import mido
from mido import MidiFile, MidiTrack, Message

def calculate_duration_from_bpm(wav_file, bpm, frequency=600, threshold=2500):
    """
    Calculate how long the 600 Hz signal lasts in a WAV file based on its BPM, with note-on/off based on a threshold.

    Args:
    - wav_file (str): Path to the WAV file.
    - bpm (float): Beats per minute of the 600 Hz signal.
    - frequency (float): Frequency of the signal to extract (default is 600 Hz).
    - threshold (float): Amplitude threshold to determine when a note is on/off.

    Returns:
    - duration (float): Duration of the signal in seconds.
    """
    # Read the WAV file
    sample_rate, audio_data = wavfile.read(wav_file)

    # Check if audio data is stereo (2 channels) or mono (1 channel)
    if len(audio_data.shape) == 2:
        # If stereo, we take one channel (e.g., left channel)
        audio_data = audio_data[:, 0]

    # Perform FFT to extract frequency components
    n = len(audio_data)
    frequencies = np.fft.fftfreq(n, d=1/sample_rate)
    fft_values = np.fft.fft(audio_data)

    # Create a mask for the 600 Hz frequency range
    def filter_frequency(frequency_range):
        mask = (frequencies >= frequency_range[0]) & (frequencies <= frequency_range[1])
        return mask

    # Mask for 600 Hz range
    mask_600Hz = filter_frequency((590, 610))  # A small range around 600 Hz

    # Create filtered FFT values for 600 Hz
    filtered_fft_600Hz = fft_values.copy()
    filtered_fft_600Hz[~mask_600Hz] = 0  # Zero out all other frequencies

    # Perform IFFT to get the time-domain signal for 600 Hz
    filtered_audio_600Hz = np.fft.ifft(filtered_fft_600Hz).real

    # Detect note-on and note-off events based on threshold
    note_on = False
    notes = []
    start_time = None

    for i in range(1, len(filtered_audio_600Hz)):
        # If the signal exceeds the threshold and note is off, start the note
        if filtered_audio_600Hz[i] > threshold and not note_on:
            start_time = i / sample_rate  # Note starts
            note_on = True
        # If the signal falls below the threshold and note is on, end the note
        elif filtered_audio_600Hz[i] < threshold and note_on:
            end_time = i / sample_rate  # Note ends
            notes.append((start_time, end_time))
            note_on = False

    # Calculate total duration of the signal in seconds
    total_duration = sum([end - start for start, end in notes])
    return total_duration, notes

def create_midi_from_bpm(wav_file, bpm, frequency=600, threshold=2500, midi_output='output.mid'):
    """
    Create a MIDI file based on a WAV file containing a 600 Hz signal with note-on/off based on threshold.
    
    Args:
    - wav_file (str): Path to the WAV file.
    - bpm (float): Beats per minute of the 600 Hz signal.
    - frequency (float): Frequency of the signal to extract (default is 600 Hz).
    - threshold (float): Amplitude threshold for note-on/off detection.
    - midi_output (str): Output path for the generated MIDI file.
    """
    # Calculate the duration of the 600 Hz signal from the WAV file
    total_duration, notes = calculate_duration_from_bpm(wav_file, bpm, frequency, threshold)

    # Create a new MIDI file with a tempo of 60 BPM (quarter note = 1 second)
    midi = MidiFile()
    track = MidiTrack()
    midi.tracks.append(track)

    # Add MIDI header with a tempo for 60 BPM (quarter note = 1 second)
    track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(60)))  # 60 BPM = 1 second per quarter note

    # Add note-on and note-off events based on the detected note times
    for start_time, end_time in notes:
        # Convert start_time and end_time to ticks
        start_ticks = int(start_time * 480)  # 480 ticks per beat (quarter note)
        end_ticks = int(end_time * 480)  # 480 ticks per beat

        # Note-on at start_time
        track.append(Message('note_on', note=60, velocity=64, time=start_ticks))  # Note 60 is Middle C
        # Note-off at end_time
        track.append(Message('note_off', note=60, velocity=64, time=end_ticks))

    # Save the MIDI file
    midi.save(midi_output)
    print(f"MIDI file saved as {midi_output} with a total duration of {total_duration:.2f} seconds.")

# Example usage:
wav_file = 'clear1.wav'  # Path to your WAV file
bpm_600Hz = 60  # Given BPM for the 600 Hz signal
threshold = 2500  # Amplitude threshold for note-on/off detection
create_midi_from_bpm(wav_file, bpm_600Hz, threshold=threshold, midi_output='output.mid')
