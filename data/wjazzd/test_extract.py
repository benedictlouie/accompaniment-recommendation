import numpy as np
import glob

DIR = "data/wjazzd/quantised_npz/*.npz"

files = sorted(glob.glob(DIR))
print(f"Found {len(files)} song files.")

# load first song
if files:
    file0 = files[0]
    data = np.load(file0, allow_pickle=True)

    sb = data["strong_beats"]
    mn = data["melody"]
    ch = data["chords"]

    print(f"\n=== Inspecting {file0} ===")
    print("Strong beats length:", len(sb))
    print("MIDI notes length:  ", len(mn))
    print("Chords length:      ", len(ch))

    print(sb[:8])
    print(mn[:32])
    print(ch[:8])

else:
    print("No NPZ files found.")