import librosa
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class Rhythm: 
    def __init__(self):
        return 
    
    def detect_notes_lengths(self, rms_vals, sr, seg_times, bpm=60, hop_size=512, win_size=1024 ):
        note_types = []

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
                def classify_note(duration, bpm):
                    beat_duration = 60 / bpm  # Duration of a quarter note in seconds
                    
                    note_types = {
                        "Whole Note": 4 * beat_duration,
                        "Half Note": 2 * beat_duration,
                        "Quarter Note": 1 * beat_duration,
                        "Eighth Note": 0.5 * beat_duration,
                        "Sixteenth Note": 0.25 * beat_duration
                    }

                    # Find the closest note type
                    closest_note = min(note_types, key=lambda note: abs(note_types[note] - duration))
                    return closest_note

                
                # Append the note information
                note_types.append(classify_note(beats_duration, bpm))

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

                def classify_rest(duration, bpm):
                    beat_duration = 60 / bpm  # Duration of a quarter note in seconds
                    
                    note_types = {
                        "Sixteenth Rest": 4 * beat_duration,
                        "Half Rest": 2 * beat_duration,
                        "Quarter Note": 1 * beat_duration,
                        "Eighth Note": 0.5 * beat_duration,
                        "Sixteenth Note": 0.25 * beat_duration
                    }

                    # Find the closest note type
                    closest_note = min(note_types, key=lambda note: abs(note_types[note] - duration))
                    return closest_note

                note_types.append(classify_rest(rest_duration, bpm))
        return note_types


    # rms_vals, sr, og_signal = perform_rms("../Audio/Songs/Twinkle_full.m4a")
    # adding in the length of the signal to seg_times
    # segs = calculate_new_notes(rms_vals, 512, sr)
    # segs += [len(og_signal)]
    # notes = detect_notes_lengths(rms_vals, sr, segs, bpm=75)
    #print(notes)