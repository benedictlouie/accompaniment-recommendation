from pathlib import Path
import numpy as np
import pypianoroll
import joblib
from sklearn.neighbors import NearestNeighbors

MAX_POLY = 8


def get_groove(melody):
    onsets = (melody != -1).astype(np.int8)
    density = np.array([onsets.mean()], dtype=np.float32)
    return np.concatenate([onsets, density])


def encode_polyphonic_fast(pr16, start, bar_steps, max_notes=8):
    seq = np.full((bar_steps, max_notes), -1, dtype=np.int8)

    for j in range(bar_steps):

        notes = np.where(pr16[start + j] > 0)[0]

        if len(notes) == 0:
            continue

        root = notes.min()
        intervals = notes - root
        intervals = intervals[:max_notes]

        seq[j, :len(intervals)] = intervals

    return seq


def encode_bass_fast(pr16, start, bar_steps):
    seq = np.full((bar_steps, 1), -1, dtype=np.int8)

    root_note = None

    for j in range(bar_steps):

        notes = np.where(pr16[start + j] > 0)[0]

        if len(notes) == 0:
            continue

        note = notes.min()

        if root_note is None:
            root_note = note

        seq[j, 0] = note - root_note

    return seq


if __name__ == "__main__":

    root = Path("data/lpd/lpd_5_cleansed/")

    features = []

    drum_loops = []
    piano_loops = []
    guitar_loops = []
    bass_loops = []

    matches = list(root.rglob("*.npz"))

    print("Dataset size:", len(matches))

    for i, npz_file in enumerate(matches):

        try:
            mt = pypianoroll.load(npz_file)
            pypianoroll.write("data/lpd/temp.mid", mt)
        except:
            continue

        step = mt.resolution // 4
        bar_steps = 16
        stride = step * bar_steps

        try:
            drums = mt.tracks[0].pianoroll
            piano = mt.tracks[1].pianoroll
            guitar = mt.tracks[2].pianoroll
            bass = mt.tracks[3].pianoroll
            strings = mt.tracks[4].pianoroll
        except:
            continue

        T = min(len(drums), len(strings), len(piano), len(guitar), len(bass))
        T = T - (T % step)

        if T < stride * 2:
            continue

        # --- Downsample to 16th notes (FAST) ---
        piano16 = piano[:T].reshape(-1, step, 128).max(axis=1)
        guitar16 = guitar[:T].reshape(-1, step, 128).max(axis=1)
        bass16 = bass[:T].reshape(-1, step, 128).max(axis=1)
        strings16 = strings[:T].reshape(-1, step, 128).max(axis=1)

        total16 = len(strings16)
        stride16 = 16

        for start in range(bar_steps, total16 - bar_steps, bar_steps):

            melody_start = start - bar_steps   # bar t-1
            target_start = start               # bar t

            # --- melody extraction (BAR t-1) ---
            melody = []

            for j in range(bar_steps):
                idx = melody_start + j
                if idx < 0 or idx >= total16:
                    melody.append(-1)
                    continue
                notes = np.where(strings16[idx] > 0)[0]
                melody.append(notes.max() if len(notes) else -1)

            melody = np.array(melody)

            feature = get_groove(melody)
            
            # --- drums (BAR t) ---
            d = drums[target_start * step : (target_start + bar_steps) * step]
            drum_loop = []
            for j in range(bar_steps):
                block = d[j * step:(j + 1) * step]

                if block.shape[0] == 0:
                    drum_loop.append(np.zeros(128))
                    continue

                drum_step = (block.max(axis=0) > 0).astype(np.uint8)
                drum_loop.append(drum_step)

            drum_loop = np.array(drum_loop, dtype=np.uint8)
            if np.all(feature == 0):
                drum_loop[:] = 0

            # --- encode other instruments (BAR t) ---
            piano_loop = encode_polyphonic_fast(piano16, target_start, bar_steps, MAX_POLY)
            guitar_loop = encode_polyphonic_fast(guitar16, target_start, bar_steps, MAX_POLY)
            bass_loop = encode_bass_fast(bass16, target_start, bar_steps)

            features.append(feature)
            drum_loops.append(drum_loop)
            piano_loops.append(piano_loop)
            guitar_loops.append(guitar_loop)
            bass_loops.append(bass_loop)

        if i % 100 == 0:
            print(f"Processed {i+1}/{len(matches)}")

    features = np.array(features, dtype=np.float32)

    drum_loops = np.array(drum_loops, dtype=np.uint8)
    piano_loops = np.array(piano_loops, dtype=np.int8)
    guitar_loops = np.array(guitar_loops, dtype=np.int8)
    bass_loops = np.array(bass_loops, dtype=np.int8)

    print("Feature shape:", features.shape)
    print("Drums:", drum_loops.shape)
    print("Piano:", piano_loops.shape)
    print("Guitar:", guitar_loops.shape)
    print("Bass:", bass_loops.shape)

    nn = NearestNeighbors(
        n_neighbors=10,
        metric="manhattan"
    )

    nn.fit(features)

    joblib.dump((nn, drum_loops), "data/lpd/drum_nn.joblib")
    joblib.dump((nn, piano_loops), "data/lpd/piano_nn.joblib")
    joblib.dump((nn, guitar_loops), "data/lpd/guitar_nn.joblib")
    joblib.dump((nn, bass_loops), "data/lpd/bass_nn.joblib")

    print("Saved NN maps.")