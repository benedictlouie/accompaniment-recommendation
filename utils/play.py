import numpy as np
from mido import Message, MidiFile, MidiTrack, MetaMessage, bpm2tempo
from utils.constants import REVERSE_ROOT_MAP, CHORD_TO_TETRAD

def npz_to_midi(song_num, chords, output_path="output.mid", bpm=80):
    """
    Convert an NPZ file with 'melody' (in data/pop/melody_chords/{song_num}.npz)
    and a provided 'chords' list into a MIDI file.
    - melody: numeric note values (each becomes a 16th note)
    - chords: list of chord names like ['C:maj', 'A:min', 'N', ...] (one per beat)
    """

    # === Load melody ===
    npz_path = f'data/pop/melody_chords/{song_num}.npz'
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

    # === Write melody (16th notes) ===
    for raw_note in melody:
        note = int(round(raw_note))
        if note < 0:
            melody_track.append(Message('note_off', note=0, velocity=0, time=sixteenth_ticks))
            continue

        melody_track.append(Message('note_on', note=note, velocity=90, time=0))
        melody_track.append(Message('note_off', note=note, velocity=64, time=sixteenth_ticks))


    # === Write chords (1 per beat) ===
    chords = ["N"] + chords
    for chord_name in chords:
        notes = CHORD_TO_TETRAD.get(chord_name, [])
        if not notes:
            # skip if it's 'N' or unknown chord
            chord_track.append(Message('note_off', note=0, velocity=0, time=tpb))
            continue
        # Note ONs
        for n in notes:
            if n == -1:
                chord_track.append(Message('note_off', note=0, velocity=0, time=tpb))
                continue
            chord_track.append(Message('note_on', note=n, velocity=70, time=0))
        # Note OFFs after one beat
        for i,n in enumerate(notes):
            if n == -1: continue
            chord_track.append(Message('note_off', note=n, velocity=64, time=(tpb if i == 0 else 0)))

    # === Save MIDI ===
    mid.save(output_path)
    print(f"✅ MIDI saved to {output_path}")
    