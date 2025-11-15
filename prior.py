import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from constants import QUALITY_SIMPLIFIER, QUALITIES, FIFTHS, FIFTHS_INDEX, FLAT_TO_SHARP
from matplotlib.colors import LogNorm

def simplify_chord(chord_str):
    if chord_str == "N" or chord_str is None:
        return "N"
    chord_str = chord_str.split('/')[0]
    root, qual = chord_str.split(':')
    if root.endswith('b'):
        root = FLAT_TO_SHARP.get(root, root)
    qual = QUALITY_SIMPLIFIER.get(qual, qual)
    if qual not in QUALITIES:
        qual = 'maj'
    return f"{root}:{qual}"


transitions = defaultdict(lambda: defaultdict(int))

for song_num in range(1, 301):
    song_num_str = f"{song_num:03d}"
    npz_path = f'pop/melody_chords/{song_num_str}.npz'
    data = np.load(npz_path, allow_pickle=True)
    
    chords = data["chords"]
    
    # Record transitions
    for i in range(len(chords) - 1):
        current_chord = simplify_chord(chords[i])
        next_chord = simplify_chord(chords[i + 1])

        # for i in range(-6, 6):
        #     chord_from = "N"
        #     chord_to = "N"
        #     if current_chord != "N":
        #         root_from, quality_from = current_chord.split(':')
        #         root_from = (FIFTHS_INDEX[root_from] + i) % 12
        #         chord_from = FIFTHS[(root_from + i) % 12] + ':' + quality_from
        #     if next_chord != "N":
        #         root_to, quality_to = next_chord.split(':')
        #         root_to = (FIFTHS_INDEX[root_to] + i) % 12
        #         chord_to = FIFTHS[(root_to + i) % 12] + ':' + quality_to
            
        #     transitions[chord_from][chord_to] += 1
        transitions[current_chord][next_chord] += 1

# Step 2: Create chord list
def chord_to_position(chord):

    if chord == 'N': return -1
    try: root, quality = chord.split(':')
    except ValueError:
        print(f"Warning: Invalid chord format '{chord}', expected 'Root:Quality'")
        return None
    
    if root.endswith('b'): root = FLAT_TO_SHARP.get(root, root)
    if root not in FIFTHS_INDEX:
        print(f"Warning: {root} not in circle of fifths")
        return None

    idx = FIFTHS_INDEX[root]
    if quality == "min":
        idx = (idx - 4) % 12 + 0.5
    return idx
    
chord_list = sorted(transitions.keys(), key=chord_to_position)
n_chords = len(chord_list)

# Step 3: Create transition matrix
transition_matrix = np.zeros((n_chords, n_chords))

for i, chord_from in enumerate(chord_list):
    total = sum(transitions[chord_from].values()) + 1
    for j, chord_to in enumerate(chord_list):
        transition_matrix[i, j] = transitions[chord_from][chord_to] / total

# Step 4: Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(
    transition_matrix + 1e-12,
    xticklabels=chord_list,
    yticklabels=chord_list,
    cmap="Blues",
    norm=LogNorm(vmin=transition_matrix.min() + 1e-6, vmax=transition_matrix.max()),
    annot=False
)
plt.xlabel("Next Chord")
plt.ylabel("Current Chord")
plt.title("Chord Transition Probability Heatmap")
plt.show()
