import librosa
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Segmentation.seg import perform_rms, calculate_new_notes

def detect_notes_lengths(rms_vals, sr, seg_times, bpm=60, hop_size=512, win_size=1024 ):
    note_frequencies = []
    

    # Convert BPM to seconds per beat
    seconds_per_beat = 60 / bpm
    for i in range(len(seg_times) - 1):
        start_sample = int((seg_times[i] / 1000) * sr/hop_size)
        end_sample = int((seg_times[i + 1] / 1000) * sr/hop_size)
        # Extract the segment (Assuming 'signal' is the full audio signal)
        segment = rms_vals[start_sample:end_sample]
        
        
        # Find the indices where the RMS values are above 0.01
        above_threshold_indices = np.where(segment > 0.02)[0]
        
        if len(above_threshold_indices) > 0:
        
            # We have a continuous segment where the RMS value is above the threshold
            # Calculate the start and end of this continuous segment
            continuous_start = above_threshold_indices[0]
            continuous_end = above_threshold_indices[-1]

            # Calculate the duration of this continuous segment in samples
            duration_samples = continuous_end - continuous_start + 1
            
            # Convert duration from samples to seconds
            duration_seconds = duration_samples * hop_size / sr
            
            # Convert duration to note length (in terms of beats)
            beats_duration = duration_seconds / seconds_per_beat
            
            # Determine the type of note based on beats_duration
            if beats_duration < 0.25:
                note_type = 'Sixteenth'
            elif beats_duration < 0.5:
                note_type = 'Eighth'
            elif beats_duration < 1.1: # most end up being really close to 1 so having it be a little over
                note_type = 'Quarter'
            elif beats_duration < 2.1:
                note_type = 'Half'
            else:
                note_type = 'Whole'
            
            # print(beats_duration, note_type)
            # Append the note information
            note_frequencies.append(note_type)

            # if the length of the note isn't the whole segment, add in a rest?
            # len_of_segment = (end_sample - start_sample) * hop_size / sr
            # rest_duration = (len_of_segment - duration_seconds)/ seconds_per_beat

            # if rest_duration < 0.2:
            #     continue
            # elif rest_duration < 0.25:
            #     rest_type = 'Sixteenth Rest'
            # elif rest_duration < 0.5:
            #     rest_type = 'Eighth Rest'
            # elif rest_duration < 1:
            #     rest_type = 'Quarter Rest'
            # elif rest_duration < 2:
            #     rest_type = 'Half Rest'
            # else:
            #     rest_type = 'Whole Rest'

            # note_frequencies.append(rest_type)

        else:
            # length of segment is length of the rest
            len_of_segment = (end_sample - start_sample) * hop_size / sr
            rest_duration = len_of_segment / seconds_per_beat
            if rest_duration < 0.25:
                rest_type = 'Sixteenth Rest'
            elif rest_duration < 0.5:
                rest_type = 'Eighth Rest'
            elif rest_duration < 1:
                rest_type = 'Quarter Rest'
            elif rest_duration < 2:
                rest_type = 'Half Rest'
            else:
                rest_type = 'Whole Rest'

            note_frequencies.append(rest_type)

    
    return note_frequencies


# rms_vals, sr, og_signal = perform_rms("../Audio/Songs/Twinkle_full.m4a")
# adding in the length of the signal to seg_times
# segs = calculate_new_notes(rms_vals, 512, sr)
# segs += [len(og_signal)]
# notes = detect_notes_lengths(rms_vals, sr, segs, bpm=75)
#print(notes)