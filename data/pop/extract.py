import mido
import numpy as np
import os

# ------------------------------
# Beat and chord reading
# ------------------------------

def read_beats(filename):
    data = np.loadtxt(filename)
    times = data[:, 0]
    strong_beats = data[:, 2].astype(int)
    return times, strong_beats


def read_chords(filename):
    starts, ends, names = [], [], []
    with open(filename, 'r') as f:
        for line in f:
            if line.strip():
                start, end, name = line.strip().split(maxsplit=2)
                starts.append(float(start))
                ends.append(float(end))
                names.append(name)
    return np.array(starts), np.array(ends), np.array(names)


# ------------------------------
# Align chords to beats
# ------------------------------

def chords_per_beat(beat_times, chord_starts, chord_ends, chord_names):
    result = np.full(len(beat_times), "N", dtype='<U10')
    chord_index = 0
    n_chords = len(chord_starts)

    for i, t in enumerate(beat_times):
        while chord_index < n_chords and chord_ends[chord_index] <= t:
            chord_index += 1
        if chord_index < n_chords and chord_starts[chord_index] <= t < chord_ends[chord_index]:
            result[i] = chord_names[chord_index]
    return result


# ------------------------------
# Generate sixteenth-note times
# ------------------------------

def generate_16th_notes_from_beats(beat_times):
    beat_times = np.asarray(beat_times)
    intervals = np.diff(beat_times) / 4
    sixteenth_times = np.concatenate([
        beat_times[:-1, None] + np.arange(4)[None, :] * intervals[:, None],
        beat_times[-1:].reshape(1, 1)
    ], axis=None)
    return sixteenth_times


# ------------------------------
# MIDI reading
# ------------------------------

def read_first_track_notes(midi_file):
    mid = mido.MidiFile(midi_file)
    track = mid.tracks[1]
    notes = []
    ongoing = {}
    current_tick = 0

    for msg in track:
        current_tick += msg.time
        if msg.type == 'note_on' and msg.velocity > 0:
            ongoing[msg.note] = current_tick
        elif msg.type in ('note_off', 'note_on') and msg.velocity == 0:
            if msg.note in ongoing:
                start_tick = ongoing.pop(msg.note)
                duration = current_tick - start_tick
                notes.append((start_tick, duration, msg.note))
    return mid, np.array(notes, dtype=int)


def get_tempo(mid):
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                return msg.tempo
    return 500000  # default 120 BPM


# ------------------------------
# Expand melody to 16th notes
# ------------------------------

def expand_notes_to_16ths(mid, notes, sixteenth_times):
    ticks_per_beat = mid.ticks_per_beat
    tempo = get_tempo(mid)

    start_ticks = notes[:, 0]
    dur_ticks = notes[:, 1]
    note_nums = notes[:, 2]

    starts_sec = np.array([mido.tick2second(t, ticks_per_beat, tempo) for t in start_ticks])
    ends_sec = np.array([mido.tick2second(t + d, ticks_per_beat, tempo) for t, d in zip(start_ticks, dur_ticks)])

    output_notes = np.full(len(sixteenth_times) - 1, -1, dtype=int)
    note_idx = 0
    num_notes = len(notes)

    for i in range(len(sixteenth_times) - 1):
        t_start, t_end = sixteenth_times[i], sixteenth_times[i + 1]
        while note_idx < num_notes and ends_sec[note_idx] <= t_start:
            note_idx += 1
        if note_idx < num_notes and starts_sec[note_idx] < t_end and ends_sec[note_idx] > t_start:
            output_notes[i] = note_nums[note_idx]
    return output_notes


# ------------------------------
# Export to NPZ
# ------------------------------

def export_melody_and_chords(strong_beats, melody_notes, chords_aligned, filename):
    """
    strong_beats: np.array [num_beats]
    melody_notes: np.array [num_16ths] (MIDI note numbers or -1)
    chords_aligned: np.array [num_beats]
    filename: output .npz file
    """
    num_beats = len(chords_aligned)
    melody = melody_notes[:num_beats * 4].reshape(-1, 4)

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.savez_compressed(
        filename,
        strong_beats=np.array(strong_beats, dtype=int),
        chords=np.array(chords_aligned),
        melody=melody
    )


# ------------------------------
# Main
# ------------------------------

if __name__ == "__main__":
    base_dir = "data/pop/POP909"
    out_dir = "data/pop/melody_chords"
    os.makedirs(out_dir, exist_ok=True)

    for num in range(1, 910):
        song_id = f"{num:03d}"
        beats_file = f"{base_dir}/{song_id}/beat_midi.txt"
        chords_file = f"{base_dir}/{song_id}/chord_midi.txt"
        midi_file = f"{base_dir}/{song_id}/{song_id}.mid"

        if not (os.path.exists(beats_file) and os.path.exists(chords_file) and os.path.exists(midi_file)):
            continue

        beat_times, strong_beats = read_beats(beats_file)
        chord_starts, chord_ends, chord_names = read_chords(chords_file)
        chords_aligned = chords_per_beat(beat_times, chord_starts, chord_ends, chord_names)
        sixteenth_times = generate_16th_notes_from_beats(beat_times)

        mid, notes = read_first_track_notes(midi_file)
        melody_notes = expand_notes_to_16ths(mid, notes, sixteenth_times)

        export_melody_and_chords(
            strong_beats,
            melody_notes,
            chords_aligned,
            filename=f"{out_dir}/{song_id}.npz"
        )

        print(f"Saved {song_id}.npz — beats={len(strong_beats)}, chords={len(chords_aligned)}, melody={len(melody_notes)}")
