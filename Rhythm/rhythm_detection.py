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
    
    def detect_notes_lengths(self, rms_vals, sr, seg_times, bpm=60, hop_size=512, win_size=1024):
        note_frequencies = []

        # Convert BPM to seconds per beat
        seconds_per_beat = bpm/60
        for i in range(len(seg_times) - 1):
            start_sample = int((seg_times[i] / 1000) * sr/hop_size)
            end_sample = int((seg_times[i + 1] / 1000) * sr/hop_size)
            segment = rms_vals[start_sample:end_sample]
            if len(segment) != 0:
                max_index = np.argmax(segment)
            else:
                max_index = start_sample
            rest_start = end_sample
            beg_rest = start_sample
            
            if max(segment) <= 0.3:
                rest_duration = ((end_sample - start_sample) * hop_size / sr )/ seconds_per_beat
                if rest_duration < 0.5:
                    rest_type = "skip"
                if rest_duration < 0.75:
                    rest_type = 'Sixteenth Rest'
                elif rest_duration < 1:
                    rest_type = 'Eighth Rest'
                elif rest_duration < 2:
                    rest_type = 'Quarter Rest'
                elif rest_duration < 3:
                    rest_type = 'Half Rest'
                else:
                    rest_type = 'Whole Rest'
                note_frequencies.append(rest_type)
                continue

            # check if the rest is in the beginning of the statement
            for k in range(len(segment)):
                if k < max_index and segment[k] > 0.3:
                    beg_rest = start_sample + k
                    break
              
            for j in range(len(segment)):
                if j > max_index and segment[j] <= 0.3:
                    rest_start = start_sample + j
                    break
            # Calculate the duration of this continuous segment in samples
            duration_samples = rest_start - beg_rest + 1
            # duration_samples = rest_start - start_sample + 1
            duration_seconds = duration_samples * hop_size / sr
            beats_duration = duration_seconds / seconds_per_beat
            # if the length of the note isn't the whole segment, add in a rest?
            rest_duration = ((end_sample - rest_start) * hop_size / sr)/ seconds_per_beat
            # rest_duration = (end_sample - rest_start)
            if rest_duration < 0.5:
                beats_duration += rest_duration
            # Determine the type of note based on beats_duration
            if beats_duration < 0.25:
                note_type = 'Sixteenth Note'
            elif beats_duration < 0.5:
                note_type = 'Eighth Note'
            elif beats_duration < 1.1: # most end up being really close to 1 so having it be a little over
                note_type = 'Quarter Note'
            elif beats_duration < 2.1:
                note_type = 'Half Note'
            else:
                note_type = 'Whole Note'
            
            # print(beats_duration, note_type)
            # Append the note information
            note_frequencies.append(note_type)

            
            # print(len_of_segment, duration_seconds, rest_duration)
            if rest_duration < 0.5:
                continue
            elif rest_duration < 0.75:
                rest_type = 'Sixteenth Rest'
            elif rest_duration < 1:
                rest_type = 'Eighth Rest'
            elif rest_duration < 2:
                rest_type = 'Quarter Rest'
            elif rest_duration < 3:
                rest_type = 'Half Rest'
            else:
                rest_type = 'Whole Rest'

            note_frequencies.append(rest_type)
        return note_frequencies

    def detect_silence(self, y, sr, seg_times, bpm = 60, hop_size=512, win_size=1024):
        note_frequencies = []
        # plt.figure(figsize=(14, 4))
        # librosa.display.waveshow(y, sr=sr)
        # plt.title('Raw Audio Signal (Waveform)')
        # plt.xlabel('Time (s)')
        # plt.ylabel('Amplitude')
        # plt.tight_layout()
        # plt.show()
        seconds_per_beat = bpm/60
        for i in range(len(seg_times) - 1):
            start_sample = int((seg_times[i] / 1000) * sr/hop_size)
            end_sample = int((seg_times[i + 1] / 1000) * sr/hop_size)
            segment = y[start_sample:end_sample]
            
            intervals = librosa.effects.split(segment, frame_length=8000, top_db=5)
            for interval in intervals:
                start, end = interval
                duration_seconds = (end - start) * hop_size / sr
                
                beats_duration = duration_seconds / seconds_per_beat
                if beats_duration < 0.25:
                    note_type = 'Sixteenth Note'
                elif beats_duration < 0.5:
                    note_type = 'Eighth Note'
                elif beats_duration < 1.1: # most end up being really close to 1 so having it be a little over
                    note_type = 'Quarter Note'
                elif beats_duration < 2.1:
                    note_type = 'Half Note'
                else:
                    note_type = 'Whole Note'
                
                # print(beats_duration, note_type)
                # Append the note information
                note_frequencies.append(note_type)
            rest_duration = (end_sample - start_sample) * hop_size/sr - duration_seconds
            print(rest_duration)
            if rest_duration < 0.5:
                continue
            elif rest_duration < 0.75:
                rest_type = 'Sixteenth Rest'
            elif rest_duration < 1:
                rest_type = 'Eighth Rest'
            elif rest_duration < 2:
                rest_type = 'Quarter Rest'
            elif rest_duration < 3:
                rest_type = 'Half Rest'
            else:
                rest_type = 'Whole Rest'

            note_frequencies.append(rest_type)
        print(note_frequencies)





    # segmentation = Segmentation()
    # rms_vals, sr, og_signal = ste("../Audio/Songs/Twinkle_full.m4a")
    # # adding in the length of the signal to seg_times
    # segs = segment_notes(rms_vals, 512, sr)
    # # segs += [len(og_signal)]
    # notes = detect_notes_lengths(rms_vals, sr, segs, bpm=75)
    # print(notes)

