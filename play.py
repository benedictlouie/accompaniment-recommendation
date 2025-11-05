import numpy as np
from mido import Message, MidiFile, MidiTrack, MetaMessage, bpm2tempo

def build_chord_map():
    """Builds a dictionary of all major/minor chords (with sharps & flats) two octaves lower."""
    notes = {
        'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
        'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8,
        'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
    }

    chord_map = {}
    base_octave = 36  # C2 (~two octaves below middle C)

    for name, offset in notes.items():
        root = base_octave + offset
        # Major chord: root, +4, +7
        chord_map[f"{name}:maj"] = [root, root + 4, root + 7]
        # Minor chord: root, +3, +7
        chord_map[f"{name}:min"] = [root, root + 3, root + 7]

    return chord_map


def npz_to_midi(song_num, chords, output_path="output.mid", bpm=80):
    """
    Convert an NPZ file with 'melody' (in pop/melody_chords/{song_num}.npz)
    and a provided 'chords' list into a MIDI file.
    - melody: numeric note values (each becomes a 16th note)
    - chords: list of chord names like ['C:maj', 'A:min', 'N', ...] (one per beat)
    """

    # === Load melody ===
    npz_path = f'pop/melody_chords/{song_num}.npz'
    data = np.load(npz_path, allow_pickle=True)
    melody = data['melody'].flatten()

    # === MIDI setup ===
    mid = MidiFile()
    melody_track = MidiTrack()
    chord_track = MidiTrack()
    mid.tracks.append(melody_track)
    mid.tracks.append(chord_track)

    # === Timing setup ===
    tpb = 480  # ticks per beat
    sixteenth_ticks = tpb // 4
    tempo = bpm2tempo(bpm)
    melody_track.append(MetaMessage('set_tempo', tempo=tempo, time=0))

    chord_map = build_chord_map()

    # === Write melody (16th notes) ===
    for raw_note in melody:
        note = int(round(raw_note))
        if note < 0:
            melody_track.append(Message('note_off', note=0, velocity=0, time=sixteenth_ticks))
            continue

        melody_track.append(Message('note_on', note=note, velocity=90, time=0))
        melody_track.append(Message('note_off', note=note, velocity=64, time=sixteenth_ticks))


    # === Write chords (1 per beat) ===
    for chord_name in chords:
        notes = chord_map.get(chord_name, [])
        if not notes:
            # skip if it's 'N' or unknown chord
            chord_track.append(Message('note_off', note=0, velocity=0, time=tpb))
            continue
        # Note ONs
        for n in notes:
            chord_track.append(Message('note_on', note=n, velocity=70, time=0))
        # Note OFFs after one beat
        for i,n in enumerate(notes):
            chord_track.append(Message('note_off', note=n, velocity=64, time=(tpb if i == 0 else 0)))

    # === Save MIDI ===
    mid.save(output_path)
    print(f"✅ MIDI saved to {output_path}")


# Example usage:
# chords_list = ["C:maj", "A:min", "F:maj", "G:maj", "N", "C:maj"]
# npz_to_midi(12, chords_list, "song12.mid", bpm=120)
