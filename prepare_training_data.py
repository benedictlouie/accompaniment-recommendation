import numpy as np
import os
import random

# ------------------------- #
#     CONSTANTS / MAPS      #
# ------------------------- #

ROOTS = np.array(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])
QUALITIES = ['maj', 'min']
CHORD_CLASSES = np.array([f"{r}:{q}" for r in ROOTS for q in QUALITIES] + ["N"])
NUM_CLASSES = len(CHORD_CLASSES)

REVERSE_ROOT_MAP = {r: i for i, r in enumerate(ROOTS)}
REVERSE_CHORD_MAP = {c: i for i, c in enumerate(CHORD_CLASSES)}
FLAT_TO_SHARP = {'Ab': 'G#', 'Bb': 'A#', 'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#'}

MEMORY = 32
MELODY_NOTES_PER_BEAT = 4


# ------------------------- #
#     CHORD UTILITIES       #
# ------------------------- #

def simplify_chord(chord_str):
    if chord_str == "N" or chord_str is None:
        return "N"
    chord_str = chord_str.split('/')[0]
    root, qual = chord_str.split(':')
    if root.endswith('b'):
        root = FLAT_TO_SHARP.get(root, root)
    qual = {'maj7': 'maj', 'min7': 'min', 'aug': 'min', 'dim': 'min', 'dim7': 'min',
            'sus2': 'maj', 'sus4': 'maj', '7': 'maj'}.get(qual, qual)
    if qual not in QUALITIES:
        qual = 'maj'
    return f"{root}:{qual}"


def chord_to_index(chord_simple):
    """Return integer index for a chord."""
    return REVERSE_CHORD_MAP.get(chord_simple, random.randint(0, NUM_CLASSES - 1))


def pad_sequence(arr, target_len, pad_value=0):
    n = len(arr)
    if n >= target_len:
        return arr[-target_len:]
    pad_shape = (target_len - n, *arr.shape[1:])
    pad = np.full(pad_shape, pad_value, dtype=arr.dtype)
    return np.concatenate([pad, arr], axis=0)


# ------------------------- #
#     SONG PREPARATION      #
# ------------------------- #

def prepare_one_song_for_training(npz_path, transpose=0):
    data = np.load(npz_path, allow_pickle=True)
    strong_beats = data["strong_beats"]
    chords = data["chords"]
    melody = data["melody"]

    # Flatten melody back into 1D
    melody = melody.flatten()

    processed_chords = []
    for chord in chords:
        chord = simplify_chord(chord)
        if chord != "N":
            root, qual = chord.split(':')
            root = ROOTS[(REVERSE_ROOT_MAP[root] + transpose) % 12]
            chord = f"{root}:{qual}"
        processed_chords.append(chord)

    chord_indices = np.array([chord_to_index(c) for c in processed_chords], dtype=np.int16)
    melody = np.where(melody > 10, melody + transpose, -1)  # transpose melody
    num_beats = min(len(chord_indices), len(melody) // MELODY_NOTES_PER_BEAT) - 1
    # if num_beats <= 0:
    #     return np.empty((0, 30 + 1)), np.empty((0, 1), dtype=np.int16)

    melody_chunks = np.reshape(melody[:num_beats * MELODY_NOTES_PER_BEAT],
                               (num_beats, MELODY_NOTES_PER_BEAT))
    strong_beats = strong_beats[:num_beats, None]
    chord_vecs = chord_indices[:num_beats, None]  # shape (num_beats, 1)
    targets = chord_indices[1:num_beats + 1, None]  # next chord index

    inputs = np.concatenate([strong_beats, melody_chunks, chord_vecs], axis=1)
    return inputs, targets


def break_down_one_song_into_sequences(song_num, test=False):
    song_num = f"{song_num:03d}"
    npz_path = f'pop/melody_chords/{song_num}.npz'

    feature_size = 1 + MELODY_NOTES_PER_BEAT + 1  # strong beat + melody + chord index

    if not os.path.exists(npz_path):
        return np.empty((0, MEMORY, feature_size)), np.empty((0, 1), dtype=np.int16)

    transpositions = [0] if test else range(-6, 6)
    all_inputs, all_targets = [], []

    for tr in transpositions:
        inputs, targets = prepare_one_song_for_training(npz_path, transpose=tr)
        if inputs.size == 0:
            continue

        seqs = []
        for i in range(inputs.shape[0]):
            seq = inputs[max(0, i - MEMORY + 1):i + 1]
            seq = pad_sequence(seq, MEMORY, pad_value=-1)
            seqs.append(seq)

        all_inputs.append(np.stack(seqs))
        all_targets.append(targets)

    if not all_inputs:
        return np.empty((0, MEMORY, feature_size)), np.empty((0, 1), dtype=np.int16)

    all_inputs = np.concatenate(all_inputs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    if not test:
        melody_part = all_inputs[:, :, 1:1 + MELODY_NOTES_PER_BEAT]
        mask = np.any(melody_part != -1, axis=(1, 2))
        all_inputs = all_inputs[mask]
        all_targets = all_targets[mask]

    return all_inputs, all_targets


# ------------------------- #
#        MAIN SCRIPT        #
# ------------------------- #

if __name__ == "__main__":
    # ---------- TRAINING DATA ----------
    train_inputs, train_targets = [], []
    for num in range(1, 301):
        X, Y = break_down_one_song_into_sequences(num)
        if X.size == 0:
            continue
        train_inputs.append(X)
        train_targets.append(Y)

    if train_inputs:
        train_inputs = np.concatenate(train_inputs, axis=0)
        train_targets = np.concatenate(train_targets, axis=0)
        print("Train:", train_inputs.shape, train_targets.shape)

        np.savez_compressed(
            "data_train.npz",
            chord_classes=CHORD_CLASSES,
            inputs=train_inputs,
            targets=train_targets
        )

    # ---------- VALIDATION DATA ----------
    val_inputs, val_targets = [], []
    for num in range(801, 901):
        X, Y = break_down_one_song_into_sequences(num, test=True)
        if X.size == 0:
            continue
        val_inputs.append(X)
        val_targets.append(Y)

    if val_inputs:
        val_inputs = np.concatenate(val_inputs, axis=0)
        val_targets = np.concatenate(val_targets, axis=0)
        print("Val:", val_inputs.shape, val_targets.shape)

        np.savez_compressed(
            "data_val.npz",
            chord_classes=CHORD_CLASSES,
            inputs=val_inputs,
            targets=val_targets
        )
