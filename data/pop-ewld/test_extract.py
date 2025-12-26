import numpy as np
import sys

# Load the NPZ file
npz_file_path = 'data/pop-ewld/melody_chords/Moonlight_Serenade.npz'
data = np.load(npz_file_path)

# Print the contents of the NPZ file
for key in data.files:
    print(f"{key}: {data[key]}")

# np.set_printoptions(threshold=sys.maxsize)
# mel = data['chords']
# for i in range(0,len(mel),3):
#     print(mel[i:i+3])