import numpy as np
import torch
import pygame
import joblib
import fluidsynth

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ROOTS = np.array(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])
QUALITIES = ['maj', 'min']
CHORD_CLASSES = np.array([f"{r}:{q}" for r in ROOTS for q in QUALITIES] + ["N"])
NUM_CLASSES = len(CHORD_CLASSES)
REVERSE_ROOT_MAP = {r: i for i, r in enumerate(ROOTS)}
REVERSE_CHORD_MAP = {c: i for i, c in enumerate(CHORD_CLASSES)}
FLAT_TO_SHARP = {'Ab': 'G#', 'Bb': 'A#', 'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#'}
QUALITY_SIMPLIFIER = {'maj7': 'maj', 'min7': 'min', 'aug': 'min', 'dim': 'min', 'dim7': 'min', 'sus2': 'maj', 'sus4': 'maj', '7': 'maj'}

QUALITIES_ALL = ['maj', 'min', 'maj7', 'min7', 'aug', 'dim', 'dim7', 'sus2', 'sus4', '7', '6', 'min6', 'm7b5', 'mM7']
QUALITY_SIMPLIFIER_REVERSE = {
    '7': ['9','11','13','79','79b','79#','7911','7911#','7913','7913b','79b13','79b13b','79#13','79#11#','7alt'],
    'maj7': ['maj9','maj79','j79#','j7911#','j79#11#'],
    'min7': ['min9','min11','min13','min79','min79b','min7911','min7913'],
    'mM7': ['mM7911#','mM7913'],
    '6': ['69','6911#'],
    'min6': ['min69'],
    'aug': ['aug7','aug79','aug79#','aug79b','aug7911#','augj7'],
    'sus2': ['power'],
    'maj': ['pedal']
}
QUALITY_SIMPLIFIER_ALL = {
    o: simplified
    for simplified, originals in QUALITY_SIMPLIFIER_REVERSE.items()
    for o in originals
}

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
    CHORD_TO_TETRAD[f"{name}:6"] = [root, root + 4, root + 7, root + 9]
    CHORD_TO_TETRAD[f"{name}:min6"] = [root, root + 3, root + 7, root + 9]
    CHORD_TO_TETRAD[f"{name}:m7b5"] = [root, root + 3, root + 6, root + 10]
    CHORD_TO_TETRAD[f"{name}:mM7"] = [root, root + 3, root + 7, root + 11]

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
TEMPERATURE = 0.3

# KRUMHANSL-KESSLER PROFILES
MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                  2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                  2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Settings
BEATS_PER_BAR = 4
STEPS_PER_BEAT = MELODY_NOTES_PER_BEAT
STEPS_PER_BAR = BEATS_PER_BAR * STEPS_PER_BEAT

SAMPLE_RATE = 44100

pygame.init()
pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=1)

# Load metronome click
CLICK_SOUND = pygame.mixer.Sound("utils/click.wav")
CLICK_SOUND_STRONG = pygame.mixer.Sound("utils/click_strong.wav")
CLICK_SOUND.set_volume(0.3)
CLICK_SOUND_STRONG.set_volume(0.3)

# Key mapping (1.5 octaves starting from C4)
KEYBOARD_MAP = {
    'a': 'C4', 'w': 'C#4', 's': 'D4', 'e': 'D#4', 'd': 'E4',
    'f': 'F4', 't': 'F#4', 'g': 'G4', 'y': 'G#4', 'h': 'A4',
    'u': 'A#4', 'j': 'B4', 'k': 'C5', 'o': 'C#5', 'l': 'D5',
    'p': 'D#5', ';': 'E5', "'": 'F5'
}
NOTE_TO_KEYBOARD = {v: k for k, v in KEYBOARD_MAP.items()}
FONT = pygame.font.SysFont(None, 24)
FONT_BIG = pygame.font.SysFont("Arial", 42)
FONT_MED = pygame.font.SysFont("Arial", 28)
FONT_SMALL = pygame.font.SysFont("Arial", 18)

BLACK = (20, 20, 20)
DARK_GRAY = (40, 44, 52)
WHITE = (240, 240, 240)
GRAY = (120, 120, 120)
BLUE = (90, 170, 255)
RED = (255, 90, 90)
GREEN = (80, 220, 140)

# Frequencies for each note
NOTE_FREQS = {
    'C2': 65.41, 'C#2': 69.30, 'D2': 73.42, 'D#2': 77.78, 'E2': 82.41,
    'F2': 87.31, 'F#2': 92.50, 'G2': 98.00, 'G#2': 103.83, 'A2': 110.00,
    'A#2': 116.54, 'B2': 123.47, 'C3': 130.81, 'C#3': 138.59, 'D3': 146.83,
    'D#3': 155.56, 'E3': 164.81, 'F3': 174.61, 'F#3': 185.00, 'G3': 196.00,
    'G#3': 207.65, 'A3': 220.00, 'A#3': 233.08, 'B3': 246.94, 'C4': 261.63,
    'C#4': 277.18, 'D4': 293.66, 'D#4': 311.13, 'E4': 329.63, 'F4': 349.23,
    'F#4': 369.99, 'G4': 392.00, 'G#4': 415.30, 'A4': 440.00, 'A#4': 466.16,
    'B4': 493.88, 'C5': 523.25, 'C#5': 554.37, 'D5': 587.33, 'D#5': 622.25,
    'E5': 659.25, 'F5': 698.46, 'F#5': 739.99, 'G5': 783.99, 'G#5': 830.61,
    'A5': 880.00, 'A#5': 932.33, 'B5': 987.77, 'C6': 1046.50, 'C#6': 1108.73,
    'D6': 1174.66, 'D#6': 1244.51, 'E6': 1318.51, 'F6': 1396.91, 'F#6': 1479.98,
    'G6': 1567.98, 'G#6': 1661.22, 'A6': 1760.00, 'A#6': 1864.66, 'B6': 1975.53
}

NOTE_TO_MIDI = {
    'C2': 36, 'C#2': 37, 'D2': 38, 'D#2': 39, 'E2': 40, 'F2': 41, 'F#2': 42,
    'G2': 43, 'G#2': 44, 'A2': 45, 'A#2': 46, 'B2': 47, 'C3': 48, 'C#3': 49,
    'D3': 50, 'D#3': 51, 'E3': 52, 'F3': 53, 'F#3': 54, 'G3': 55, 'G#3': 56,
    'A3': 57, 'A#3': 58, 'B3': 59, 'C4': 60, 'C#4': 61, 'D4': 62, 'D#4': 63,
    'E4': 64, 'F4': 65, 'F#4': 66, 'G4': 67, 'G#4': 68, 'A4': 69, 'A#4': 70,
    'B4': 71, 'C5': 72, 'C#5': 73, 'D5': 74, 'D#5': 75, 'E5': 76, 'F5': 77,
    'F#5': 78, 'G5': 79, 'G#5': 80, 'A5': 81, 'A#5': 82, 'B5': 83, 'C6': 84,
    'C#6': 85, 'D6': 86, 'D#6': 87, 'E6': 88, 'F6': 89, 'F#6': 90, 'G6': 91,
    'G#6': 92, 'A6': 93, 'A#6': 94, 'B6': 95
}

NN, DRUMS = joblib.load("data/lpd/drum_nn.joblib")
_, BASSES = joblib.load("data/lpd/bass_nn.joblib")
_, GUITARS = joblib.load("data/lpd/guitar_nn.joblib")
_, PIANOS = joblib.load("data/lpd/piano_nn.joblib")
