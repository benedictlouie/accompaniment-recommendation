import os
import music21
import mido
import math
from mido import MidiFile, MidiTrack, Message, MetaMessage
import numpy as np

# Define the ticks per quarter note (standard is 480 ticks per quarter note)
TICKS_PER_QUARTER_NOTE = 480

def convert_mxl_to_midi(mxl_file_path, midi_output_path):
    # Parse the MusicXML file using music21
    score = music21.converter.parse(mxl_file_path)
    midi_file = MidiFile()

    # Create two tracks for melody and chords
    melody_track = MidiTrack()
    chord_track = MidiTrack()

    # Add both tracks to the MIDI file
    midi_file.tracks.append(melody_track)
    midi_file.tracks.append(chord_track)

    time_signature = score.flatten().getElementsByClass(music21.meter.TimeSignature)
    if time_signature:
        time_signature = time_signature[0]
    else:
        time_signature = music21.meter.TimeSignature('4/4')
        
    numerator = time_signature.numerator
    denominator = time_signature.denominator

    # Add time signature meta message at the beginning of the melody track
    melody_track.append(
        MetaMessage(
            'time_signature',
            numerator=numerator,
            denominator=denominator,
            clocks_per_click=24,
            notated_32nd_notes_per_beat=8,
            time=0
        )
    )

    # Iterate through parts and process notes and chords
    for part in score.parts:
        melody_notes = []  # List to store melody notes for simultaneous playback

        last_chord = []

        # Iterate through measures
        for measure in part.getElementsByClass(music21.stream.Measure):
            measure_time = 0  # Reset the measure time for each measure

            # Collect all events (notes, chords, and rests) in the measure
            events = []
            has_chord_in_measure = False  # Flag to check if there are any chords in the measure
            first_chord_in_measure = float('inf')

            for element in measure.flatten().notesAndRests:  # Use notesAndRests to include rests
                events.append((element, measure_time))
                if isinstance(element, music21.note.Note):
                    measure_time += int(element.quarterLength * TICKS_PER_QUARTER_NOTE)
                elif isinstance(element, music21.chord.Chord):
                    has_chord_in_measure = True  # We found a chord
                    first_chord_in_measure = min(first_chord_in_measure, measure_time)
                elif isinstance(element, music21.note.Rest):
                    measure_time += int(element.quarterLength * TICKS_PER_QUARTER_NOTE)

            # Now process events to create MIDI messages
            for i, (element, start_time) in enumerate(events):
                # If this is a note, add it to the melody track
                if isinstance(element, music21.note.Note):
                    if element.pitch:
                        pitch = int(element.pitch.ps)  # MIDI pitch number for melody
                        # Get the actual note duration in ticks
                        duration_in_ticks = int(element.quarterLength * TICKS_PER_QUARTER_NOTE)

                        # Add melody note (note_on and note_off with correct timing)
                        melody_track.append(Message('note_on', note=pitch, velocity=64, time=0))
                        melody_track.append(Message('note_off', note=pitch, velocity=64, time=duration_in_ticks))

                        # Update the current time and last note end time
                        melody_notes.append((pitch, duration_in_ticks))

                # If this is a chord, calculate how long it lasts until the next chord
                elif isinstance(element, music21.chord.Chord):
                    chord_pitches = [int(note.pitch.ps) for note in element.notes if note.pitch]  # Lower each note by an octave
                    if len(chord_pitches) > 0:  # Only process if chord has notes
                        # Find the next chord's start time (or the end of the current measure if no more chords)
                        next_chord_start_time = measure_time
                        if i + 1 < len(events):  # If there is a next event
                            for j in range(i + 1, len(events)):
                                if isinstance(events[j][0], music21.chord.Chord):
                                    next_chord_start_time = events[j][1]
                                    break
                                
                        # Calculate chord duration as the time until the next chord
                        chord_duration_in_ticks = max(0, next_chord_start_time - start_time)

                        # Play the chord (all notes at the same time) on the chord track
                        for idx, pitch in enumerate(chord_pitches):
                            chord_track.append(Message('note_on', note=pitch, velocity=64, time=start_time if start_time == first_chord_in_measure and idx == 0 else 0)) 
                        for idx, pitch in enumerate(chord_pitches):
                            chord_track.append(
                                Message('note_off', note=pitch, velocity=64, time=chord_duration_in_ticks if idx == 0 else 0)
                            )
                        last_chord = chord_pitches

                # If it's a rest, add it as a "no note" event
                elif isinstance(element, music21.note.Rest):
                    rest_duration_in_ticks = int(element.quarterLength * TICKS_PER_QUARTER_NOTE)  # Convert to ticks
                    melody_track.append(Message('note_off', note=0, velocity=0, time=rest_duration_in_ticks))  # Rest duration

            # If the measure had no chords, add rest events in the chord track
            if not has_chord_in_measure:
                if last_chord:
                    chord_duration_in_ticks = measure_time
                    for pitch in last_chord:
                        chord_track.append(Message('note_on', note=pitch, velocity=64, time=0))  # Chord note on
                    for idx, pitch in enumerate(last_chord):
                        chord_track.append(
                            Message('note_off', note=pitch, velocity=64, time=chord_duration_in_ticks if idx == 0 else 0)
                        )
                else:
                    rest_duration_in_ticks = measure_time
                    chord_track.append(Message('note_off', note=0, velocity=0, time=0))  # Rest at the beginning
                    chord_track.append(Message('note_off', note=0, velocity=0, time=rest_duration_in_ticks))  # Rest for the whole measure

    # Save the MIDI file
    midi_file.save(midi_output_path)


import numpy as np
from music21 import *

TICKS_PER_QUARTER_NOTE = 480  # Define the number of ticks per quarter note
TICKS_PER_16TH_NOTE = TICKS_PER_QUARTER_NOTE // 4  # 1/16th note duration in ticks

def extract_data_from_mxl_to_npz(mxl_file_path, npz_output_path):
    # Parse the MusicXML file using music21
    score = music21.converter.parse(mxl_file_path)

    # Initialize empty lists to store the data
    melody = []
    chord_symbols = []
    strong_beats = []

    # To keep track of the last chord
    last_chord = 'N'

    # Iterate over all parts in the score
    for part in score.parts:
        # Get the time signature (first measure or default to 4/4 if not found)
        time_signature = part.flatten().getElementsByClass(music21.meter.TimeSignature)
        if time_signature:
            time_signature = time_signature[0]
        else:
            time_signature = music21.meter.TimeSignature('4/4')
        numerator = time_signature.numerator
        denominator = time_signature.denominator
        if denominator not in [4,8]:
            return

        # Iterate over all measures in the part
        for measure in part.getElementsByClass(music21.stream.Measure):
            # Initialize the strong beats array for this measure (0 for all beats initially)
            measure_strong_beats = [1] + [0] * (numerator-1)
            strong_beats.extend(measure_strong_beats)  # Append to the strong beats list

            # Collect all events (notes, chords, and rests) in the measure
            melody_in_bar = []
            chords_in_bar = []

            for element in measure.flatten().notesAndRests:

                # how many time-steps this element lasts (16th-note resolution)
                reps = int(element.duration.quarterLength / 0.25)

                if isinstance(element, music21.note.Note):
                    for _ in range(reps):
                        melody_in_bar.append(element.pitch.ps)

                elif isinstance(element, music21.chord.Chord):
                    chord_symbol = element.figure
                    while len(chords_in_bar) < int(element.offset * denominator / 4):
                        chords_in_bar.append(last_chord)

                    chords_in_bar.append(chord_symbol)
                    last_chord = chord_symbol

                elif isinstance(element, music21.note.Rest):
                    for _ in range(reps):
                        melody_in_bar.append(-1)

            while len(melody_in_bar) < numerator * 4:
                melody_in_bar.append(-1)
            melody += melody_in_bar[:numerator * 4]

            while len(chords_in_bar) < numerator:
                chords_in_bar.append(last_chord) 

            chord_symbols += chords_in_bar[:numerator]

    # Pad the melody list to match the length of the strong beats array
    # Ensuring that melody and strong_beats are of the same length (important for NPZ)
    max_length = len(strong_beats)
    melody = np.pad(melody, (0, max(0, max_length - len(melody))), constant_values=-1)

    # Convert lists to numpy arrays
    melody = np.array(melody)
    melody = np.repeat(melody, denominator / 4)

    chord_symbols = np.array(chord_symbols)
    strong_beats = np.array(strong_beats)

    # Save the data as a .npz file
    np.savez(npz_output_path, 
             filename=mxl_file_path, 
             strong_beats=strong_beats, 
             chords=chord_symbols, 
             melody=melody)



if __name__ == "__main__":

    input_directory = 'data/pop-ewld/ewld/'  # Replace with your directory containing .mxl files
    midi_output_directory = 'data/pop-ewld/midi_files/'  # Change this to your desired MIDI output directory
    npz_output_directory = 'data/pop-ewld/melody_chords/'  # Change this to your desired .npz output directory

    if not os.path.exists(midi_output_directory):
        os.makedirs(midi_output_directory)

    if not os.path.exists(npz_output_directory):
        os.makedirs(npz_output_directory)

    # Loop through all .mxl files in the input directory and its subdirectories
    for root, dirs, files in os.walk(input_directory):  # Walk through all subdirectories
        for filename in files:
            if filename.endswith(".mxl"):
                # Get the full path to the .mxl file
                mxl_file_path = os.path.join(root, filename)

                # Generate MIDI output file path (no subfolders, just save directly under melody_chords)
                midi_output_path = os.path.join(midi_output_directory, filename.replace('.mxl', '.mid'))

                # Generate .npz output file path (no subfolders, just save directly under melody_chords)
                npz_output_path = os.path.join(npz_output_directory, filename.replace('.mxl', '.npz'))

                # Create the necessary subdirectories in output if they don't exist
                os.makedirs(os.path.dirname(midi_output_path), exist_ok=True)
                os.makedirs(os.path.dirname(npz_output_path), exist_ok=True)

                # Convert to MIDI
                convert_mxl_to_midi(mxl_file_path, midi_output_path)

                # Extract data and save as .npz
                extract_data_from_mxl_to_npz(mxl_file_path, npz_output_path)
