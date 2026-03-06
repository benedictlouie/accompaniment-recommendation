from pathlib import Path
import numpy as np
import pypianoroll
import mido
import joblib
from sklearn.neighbors import NearestNeighbors

def get_groove(melody):
    onsets = (melody != -1).astype(np.int8)
    density = np.array([onsets.mean()], dtype=np.float32)
    return np.concatenate([onsets, density])


if __name__ == "__main__":

    root = Path("data/lpd/lpd_5_cleansed/")

    features = []
    drum_loops = []
    matches = root.rglob("*.npz")
    matches = list(matches)
    print("Dataset size:", len(matches))

    for i, npz_file in enumerate(matches):
        try:
            mt = pypianoroll.load(npz_file)
        except Exception as e:
            print(f"Error loading {npz_file}: {e}")
            continue

        step = mt.resolution // 4   # ticks per 16th note
        
        # Get time signature
        beats_per_bar = 4
        bar_steps = beats_per_bar * 4
        stride = step * bar_steps

        drums = mt.tracks[0].pianoroll
        strings = mt.tracks[4].pianoroll

        T = min(len(drums), len(strings))
        T = T - (T % stride)

        for start in range(stride, T - stride, stride):
            d = drums[start + stride : start + 2 * stride]
            s = strings[start - step : start + stride - step]

            # --- melody (top note per 16th note) ---
            melody = []
            for j in range(bar_steps):
                block = s[j*step:(j+1)*step]
                notes = np.where(block.max(axis=0) > 0)[0]
                melody.append(notes.max() if len(notes) else -1)
            melody = np.array(melody)

            # --- feature ---
            feature = get_groove(melody)

            # --- drum loop (binary) downsampled to 16th notes ---
            drum_loop = []
            for j in range(bar_steps):
                block = d[j*step:(j+1)*step]
                drum_step = (block.max(axis=0) > 0).astype(np.uint8)
                drum_loop.append(drum_step)
            drum_loop = np.array(drum_loop, dtype=np.uint8)

            if np.all(feature == 0):
                drum_loop[:] = 0

            features.append(feature)
            drum_loops.append(drum_loop)
        
        if i % 100 == 0:
            print(f"Extracted {i+1} out of {len(matches)}")

    features = np.array(features, dtype=np.int8)          # shape: (N, 15)
    drum_loops = np.array(drum_loops, dtype=np.uint8)     # shape: (N,16,128)

    print("Shape of features:", features.shape)
    print("Shape of drum_loops:", drum_loops.shape)

    drum_nn = NearestNeighbors(
        n_neighbors=10,
        metric="manhattan"
    )
    drum_nn.fit(features)
    joblib.dump((drum_nn, drum_loops), "data/lpd/drum_nn.joblib")