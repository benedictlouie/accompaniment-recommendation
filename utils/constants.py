import numpy as np
import torch
import pygame

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ROOTS = np.array(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])
QUALITIES = ['maj', 'min']
CHORD_CLASSES = np.array([f"{r}:{q}" for r in ROOTS for q in QUALITIES] + ["N"])
NUM_CLASSES = len(CHORD_CLASSES)
REVERSE_ROOT_MAP = {r: i for i, r in enumerate(ROOTS)}
REVERSE_CHORD_MAP = {c: i for i, c in enumerate(CHORD_CLASSES)}
FLAT_TO_SHARP = {'Ab': 'G#', 'Bb': 'A#', 'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#'}
QUALITY_SIMPLIFIER = {'maj7': 'maj', 'min7': 'min', 'aug': 'min', 'dim': 'min', 'dim7': 'min', 'sus2': 'maj', 'sus4': 'maj', '7': 'maj'}

QUALITIES_ALL = ['maj', 'min', 'maj7', 'min7', 'aug', 'dim', 'dim7', 'sus2', 'sus4', '7']
CHORD_CLASSES_ALL = np.array([f"{r}:{q}" for r in ROOTS for q in QUALITIES_ALL] + ["N"])
NUM_CLASSES_ALL = len(CHORD_CLASSES_ALL)
REVERSE_CHORD_MAP_ALL = {c: i for i, c in enumerate(CHORD_CLASSES_ALL)}

# ENCODING CHORDS
CHORD_TO_TETRAD = {'N': [-1, -1, -1, -1]}
BASE_OCTAVE = 36  # C2 (~two octaves below middle C)
for name, offset in REVERSE_ROOT_MAP.items():
    root = BASE_OCTAVE + offset
    CHORD_TO_TETRAD[f"{name}:maj"] = [root, root + 4, root + 7, root + 12]
    CHORD_TO_TETRAD[f"{name}:min"] = [root, root + 3, root + 7, root + 12]
    CHORD_TO_TETRAD[f"{name}:maj7"] = [root, root + 4, root + 7, root + 11]
    CHORD_TO_TETRAD[f"{name}:min7"] = [root, root + 3, root + 7, root + 10]
    CHORD_TO_TETRAD[f"{name}:aug"] = [root, root + 4, root + 8, root + 12]
    CHORD_TO_TETRAD[f"{name}:dim"] = [root, root + 3, root + 6, root + 12]
    CHORD_TO_TETRAD[f"{name}:dim7"] = [root, root + 3, root + 6, root + 9]
    CHORD_TO_TETRAD[f"{name}:sus2"] = [root, root + 2, root + 7, root + 12]
    CHORD_TO_TETRAD[f"{name}:sus4"] = [root, root + 5, root + 7, root + 12]
    CHORD_TO_TETRAD[f"{name}:7"] = [root, root + 4, root + 7, root + 10]

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
TEMPERATURE = 0.1

# KRUMHANSL-KESSLER PROFILES
MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                  2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                  2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


# Settings
BPM = 80
BEAT_DURATION = 60 / BPM
BEATS_PER_BAR = 4
BAR_DURATION = BEAT_DURATION * BEATS_PER_BAR

pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=1)

# Load metronome click
CLICK_SOUND = pygame.mixer.Sound("utils/click.wav")
CLICK_SOUND_STRONG = pygame.mixer.Sound("utils/click_strong.wav")

# Key mapping (1.5 octaves starting from C4)
KEYBOARD_MAP = {
    'a': 'C4', 'w': 'C#4', 's': 'D4', 'e': 'D#4', 'd': 'E4',
    'f': 'F4', 't': 'F#4', 'g': 'G4', 'y': 'G#4', 'h': 'A4',
    'u': 'A#4', 'j': 'B4', 'k': 'C5', 'o': 'C#5', 'l': 'D5',
    'p': 'D#5', ';': 'E5', "'": 'F5'
}
NOTE_TO_KEYBOARD = {v: k for k, v in KEYBOARD_MAP.items()}
FONT = pygame.font.SysFont(None, 24)

# Frequencies for each note
NOTE_FREQS = {
    'C4': 261.63, 'C#4': 277.18, 'D4': 293.66, 'D#4': 311.13, 'E4': 329.63,
    'F4': 349.23, 'F#4': 369.99, 'G4': 392.00, 'G#4': 415.30, 'A4': 440.00,
    'A#4': 466.16, 'B4': 493.88, 'C5': 523.25, 'C#5': 554.37, 'D5': 587.33,
    'D#5': 622.25, 'E5': 659.25, 'F5': 698.46
}

NOTE_TO_MIDI = {
    'C4': 60, 'C#4': 61, 'D4': 62, 'D#4': 63, 'E4': 64,
    'F4': 65, 'F#4': 66, 'G4': 67, 'G#4': 68, 'A4': 69,
    'A#4': 70, 'B4': 71, 'C5': 72, 'C#5': 73, 'D5': 74,
    'D#5': 75, 'E5': 76, 'F5': 77
}
