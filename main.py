from Rhythm.rhythm import detect_notes_lengths
from Pitch.pitch import detect_notes, freq_to_note
from Segmentation.seg import perform_rms, calculate_new_notes

import pretty_midi
from music21 import converter

rms_vals, sr, og_signal = perform_rms("Audio/Songs/hotcross_interface.m4a")
segs = calculate_new_notes(rms_vals, 512, sr)
segs += [len(og_signal)]
detected_frequencies = detect_notes(og_signal, sr, segs)
detected_notes = [freq_to_note(f) for f in detected_frequencies] #list of tuples of (note_num, note)

# from rhythm
note_lengths = detect_notes_lengths(rms_vals, sr, segs, bpm=75)

print(len(note_lengths), len(detected_notes)) # need to add rests in pitch??
# print(note_lengths)

bpm = 120  #can be adjusted
seconds_per_beat = 60 / bpm  

# for now, I'm using a dictionary to keep track of the note durations. ideally, once we have the rhythm detection done,
# we want the exact note onset and offset times, which we can directly put into the MIDI calls. 
note_durations = {"whole": 4 * seconds_per_beat, "half": 2 * seconds_per_beat, "quarter": 1 * seconds_per_beat, "eighth": 0.5 * seconds_per_beat,
                  "sixteenth": 0.25 * seconds_per_beat,
}

midi = pretty_midi.PrettyMIDI()
instrument = pretty_midi.Instrument(program=74)  #flute is 74

start_time = 0

for pitch, note_type in detected_notes:
    duration = note_durations.get(note_type, 0.5)  # Default to quarter note 
    note = pretty_midi.Note(velocity=100, pitch=pitch, start=start_time, end=start_time + duration)
    instrument.notes.append(note)
    start_time += duration  

midi.instruments.append(instrument)

midi.write("melody.mid")
