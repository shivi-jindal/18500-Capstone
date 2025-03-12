import mido
from mido import MidiFile, MidiTrack, Message

# Create a new MIDI file
midi = MidiFile()

# Create a new track
track = MidiTrack()
midi.tracks.append(track)

# Add a 'program change' to select the instrument (e.g., Piano)
track.append(Message('program_change', program=0))  # 0 is for Acoustic Grand Piano

# Add a note-on event (note starts)
track.append(Message('note_on', note=60, velocity=64, time=0))  # note 60 is Middle C, velocity=64, time=0

# Add a note-off event (note stops)
track.append(Message('note_off', note=60, velocity=64, time=480))  # note 60, velocity=64, time=480 (after a duration of 480 ticks)

# Add more notes if needed
track.append(Message('note_on', note=62, velocity=64, time=0))  # note 62 is D, time=0 means immediately after previous
track.append(Message('note_off', note=62, velocity=64, time=480))  # 480 ticks duration

# Save the MIDI file
midi.save('example.mid')

print("MIDI file created successfully!")
