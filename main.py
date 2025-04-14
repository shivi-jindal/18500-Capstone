from Denoising.noise_reduction import Denoising
from Segmentation.seg import Segmentation
from Rhythm.rhythm_detection import Rhythm
from Pitch.pitch import Pitch

import pretty_midi

INPUT_FILE_NAME = "Audio/Songs/staccato_scale_phoebe.m4a"
BPM = 60  #can be adjusted
SECONDS_PER_BEAT = 60 / BPM  

#initialize all required classes
denoising = Denoising()
segmentation = Segmentation()
rhythm = Rhythm()
pitch = Pitch()
midi = pretty_midi.PrettyMIDI()

#pass the signal through a bandpass filter
y_filtered, sr = denoising.noise_suppression_pipeline(INPUT_FILE_NAME)

#do the note segmentation
rms_vals, sr, og_signal, segs = segmentation.segment_notes(y_filtered, sr, BPM)

#get the note types
note_types = rhythm.detect_notes_lengths(rms_vals, sr, segs, BPM)

#do pitch detection
detected_frequencies = pitch.detect_notes(og_signal, sr, segs)
detected_notes = [pitch.freq_to_note(f) for f in detected_frequencies] #list of note_nums

# mapping from note type to duration
note_durations = {"Whole Note": 4 * SECONDS_PER_BEAT, "Half Note": 2 * SECONDS_PER_BEAT, "Quarter Note": 1 * SECONDS_PER_BEAT, 
                  "Eighth Note": 0.5 * SECONDS_PER_BEAT, "Sixteenth": 0.25 * SECONDS_PER_BEAT, "Whole Rest": 4 * SECONDS_PER_BEAT, 
                  "Half Rest": 2 * SECONDS_PER_BEAT, "Quarter Rest": 1 * SECONDS_PER_BEAT, "Eighth Rest": 0.5 * SECONDS_PER_BEAT, 
                  "Sixteenth Rest": 0.25 * SECONDS_PER_BEAT}

instrument = pretty_midi.Instrument(program=73) 

start_time = 0
for i in range(len(detected_notes)):
    note_num = detected_notes[i]
    note_type = note_types[i]
    duration = note_durations.get(note_type, SECONDS_PER_BEAT)
    if "rest" not in note_type:
        note = pretty_midi.Note(velocity=100, pitch=note_num, start=start_time, end=start_time + duration)
        instrument.notes.append(note)
    start_time += duration  

midi.instruments.append(instrument)

midi.write("melody.mid")
