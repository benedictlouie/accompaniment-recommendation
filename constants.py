import numpy as np
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ROOTS = np.array(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])
QUALITIES = ['maj', 'min']
CHORD_CLASSES = np.array([f"{r}:{q}" for r in ROOTS for q in QUALITIES] + ["N"])
NUM_CLASSES = len(CHORD_CLASSES)
REVERSE_ROOT_MAP = {r: i for i, r in enumerate(ROOTS)}
REVERSE_CHORD_MAP = {c: i for i, c in enumerate(CHORD_CLASSES)}
FLAT_TO_SHARP = {'Ab': 'G#', 'Bb': 'A#', 'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#'}
QUALITY_SIMPLIFIER = {'maj7': 'maj', 'min7': 'min', 'aug': 'min', 'dim': 'min', 'dim7': 'min', 'sus2': 'maj', 'sus4': 'maj', '7': 'maj'}

# ENCODING CHORDS
CHORD_TO_TETRAD = {'N': [-1, -1, -1, -1]}
BASE_OCTAVE = 36  # C2 (~two octaves below middle C)
for name, offset in REVERSE_ROOT_MAP.items():
    root = BASE_OCTAVE + offset
    CHORD_TO_TETRAD[f"{name}:maj"] = [root, root + 4, root + 7, root + 12]
    CHORD_TO_TETRAD[f"{name}:min"] = [root, root + 3, root + 7, root + 12]

# INPUT DIMENSIONS
MEMORY = 32
MELODY_NOTES_PER_BEAT = 4
CHORD_EMBEDDING_LENGTH = 0
INPUT_DIM = 1 + MELODY_NOTES_PER_BEAT + CHORD_EMBEDDING_LENGTH

# CYCLE OF FIFTHS
FIFTHS = ["C", "G", "D", "A", "E", "B", "F#", "C#", "G#", "D#", "A#", "F"]
FIFTHS_INDEX = {note: i for i, note in enumerate(FIFTHS)}

def chord_to_fifths_position(chord):
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

FIFTHS_CHORD_LIST = sorted(CHORD_CLASSES, key=chord_to_fifths_position)
FIFTHS_CHORD_INDICES = {v: i for i,v in enumerate(FIFTHS_CHORD_LIST)}

# HYPERPARAMETERS
LEARNING_RATE = 1e-4
DROPOUT_RATE = 0.2
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 64
TEMPERATURE = 3

# KRUMHANSL-KESSLER PROFILES
MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                  2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                  2.54, 4.75, 3.98, 2.69, 3.34, 3.17])