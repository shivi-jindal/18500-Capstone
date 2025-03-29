from Rhythm.rhythm import detect_notes_lengths
from Pitch.pitch import detect_notes, freq_to_note
from Segmentation.seg import perform_rms, calculate_new_notes

rms_vals, sr, og_signal = perform_rms("Audio/Songs/Twinkle_full.m4a")
segs = calculate_new_notes(rms_vals, 512, sr)
segs += [len(og_signal)]
detected_frequencies = detect_notes(og_signal, sr, segs)
detected_notes = [freq_to_note(f) for f in detected_frequencies] #list of tuples of (note_num, note)

# from rhythm
note_lengths = detect_notes_lengths(rms_vals, sr, segs, bpm=75)

print(len(note_lengths), len(detected_notes))
