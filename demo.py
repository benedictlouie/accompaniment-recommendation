import pygame
import numpy as np
import torch
import time
from chord_melody_relation import predict_chords
from constants import REVERSE_ROOT_MAP, CHORD_CLASSES, NUM_CLASSES, CHORD_TO_TETRAD, FIFTHS_CHORD_LIST, FIFTHS_CHORD_INDICES, TEMPERATURE
from crf import key_probs

# Initialize pygame
pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=1)

# Settings
BPM = 80
BEAT_DURATION = 60 / BPM
BEATS_PER_BAR = 4
BAR_DURATION = BEAT_DURATION * BEATS_PER_BAR

# Load metronome click
CLICK_SOUND = pygame.mixer.Sound("click.wav")

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

# Pre-generate pygame Sounds for all notes (1s long)
def generate_note_sound(freq, duration=1.0):
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = 0.3 * np.sin(2 * np.pi * freq * t)
    wave = np.int16(wave * 32767)
    return pygame.sndarray.make_sound(np.column_stack([wave, wave]))

NOTE_SOUNDS = {note: generate_note_sound(freq*2) for note, freq in NOTE_FREQS.items()}

# Initialize mixer
pygame.mixer.init(frequency=44100, size=-16, channels=1)

# Ensure enough channels for all notes
pygame.mixer.set_num_channels(len(NOTE_FREQS) + 4)

# Channels for each note
NOTE_CHANNELS = {note: pygame.mixer.Channel(i) for i, note in enumerate(NOTE_FREQS)}

# Piano window
WIDTH, HEIGHT = 800, 200
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Piano Keyboard")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (50, 50, 50)
BLUE = (0, 0, 255)

# Key dimensions
WHITE_KEYS = ['C', 'D', 'E', 'F', 'G', 'A', 'B'] * 2
WHITE_KEY_WIDTH = WIDTH // len(WHITE_KEYS)
WHITE_KEY_HEIGHT = HEIGHT
BLACK_KEY_WIDTH = WHITE_KEY_WIDTH // 2
BLACK_KEY_HEIGHT = HEIGHT * 0.6
BLACK_KEYS = ['C#', 'D#', '', 'F#', 'G#', 'A#', '', 'C#', 'D#', '', 'F#', 'G#', 'A#', '']

# Track notes and times
notes_pressed = {}  # note -> press_time
notes_played = []   # list of (note, start_time, end_time)

# Metronome timer
last_beat_time = time.time()
current_beat = 0
bar_start_time = last_beat_time

def next_chord(bar_history, delta):

    bars = bar_history[-1]
    probs, _ = predict_chords([bars])
    probs = np.array(probs)

    num_steps, num_chords = probs.shape
    log_probs = np.log(probs + 1e-12)

    transition_matrix = np.load("chord_transition_matrix.npy") # (25, 25)
    transitions = np.sum(transition_matrix, axis=2) + 1e-12
    transitions /= transitions.sum(axis=1)
    log_transitions = np.log(transitions) * 0.3

    key_prob = key_probs(bar_history)
    rearrange = np.array([FIFTHS_CHORD_INDICES[CHORD_CLASSES[i]]-1 for i in range(NUM_CLASSES-1)])
    key_prob = key_prob[np.argsort(rearrange)]
    print("Predicted key:", FIFTHS_CHORD_LIST[np.argmax(key_prob)+1])
    probs2 = np.sum(transition_matrix, axis=1) * key_prob
    probs2 = np.sum(probs2, axis=1).flatten() + 1e-12
    probs2 /= np.sum(probs2)
    log_probs2 = np.log(probs2)
    
    return bar_history[1:, :], np.max(delta + log_probs + log_probs2 + log_transitions, axis=0)

def play_chord(bar_proportions, bar_history, delta):
    bars = [0 for _ in range(13)]
    for note, prop in bar_proportions.items():
        note = note[:-1]
        bars[REVERSE_ROOT_MAP[note]+1] += prop
    if sum(bars) > 1: return
    else: bars[0] = 1 - sum(bars)
    bars = np.array(bars) / np.sum(bars)
    bar_history = np.vstack((bar_history, bars))
    bar_history, probs = next_chord(bar_history, delta)

    decision = FIFTHS_CHORD_LIST[np.argmax(probs)]
    print("Predicted chord:", decision)
    if decision == 'N': return
    for i, midi in enumerate(CHORD_TO_TETRAD[decision]):
        wave = generate_note_sound(440 * 2 ** (1+(midi-69)/12), BAR_DURATION)
        pygame.mixer.Channel(len(NOTE_FREQS) + i).play(wave)
    return bar_history


# Main loop
running = True
while running:
    current_time = time.time()
    delta = np.zeros((1, NUM_CLASSES))
    bar_history = np.zeros((0, 13))

    # Metronome tick
    if current_time - last_beat_time >= BEAT_DURATION:

        CLICK_SOUND.play()
        last_beat_time = current_time
        current_beat += 1
        
        if current_beat >= BEATS_PER_BAR:
            # End of bar: compute proportions
            bar_end_time = current_time
            bar_proportions = {}
            for note, start, end in notes_played:
                # Clip to bar duration
                start = max(start, bar_start_time)
                end = min(end, bar_end_time)
                duration = max(0, end - start)
                if note in bar_proportions:
                    bar_proportions[note] += duration
                else:
                    bar_proportions[note] = duration
            for note in bar_proportions:
                bar_proportions[note] /= BAR_DURATION

            bar_history = play_chord(bar_proportions, bar_history, delta)
            print("Previous bar:", bar_proportions)
            print("STRONG BEAT")
            notes_played.clear()
            current_beat = 0
            bar_start_time = current_time
    
    # Draw keys
    SCREEN.fill(GRAY)
    
    # White keys
    for i, key in enumerate(WHITE_KEYS):
        note_name = key + '4' if i < 7 else key + '5'
        color = BLUE if note_name in notes_pressed else WHITE
        pygame.draw.rect(SCREEN, color, (i * WHITE_KEY_WIDTH, 0, WHITE_KEY_WIDTH, WHITE_KEY_HEIGHT))
        pygame.draw.rect(SCREEN, BLACK, (i * WHITE_KEY_WIDTH, 0, WHITE_KEY_WIDTH, WHITE_KEY_HEIGHT), 2)
        x = i * WHITE_KEY_WIDTH
        if note_name in NOTE_TO_KEYBOARD:
            text_surface = FONT.render(NOTE_TO_KEYBOARD[note_name], True, BLACK)
            text_rect = text_surface.get_rect(center=(x + WHITE_KEY_WIDTH/2, WHITE_KEY_HEIGHT - 20))
            SCREEN.blit(text_surface, text_rect)
    
    # Black keys
    for i, key in enumerate(BLACK_KEYS):
        if key != '':
            note_name = key + '4' if i < 7 else key + '5'
            color = BLUE if note_name in notes_pressed else BLACK
            pygame.draw.rect(SCREEN, color, (i * WHITE_KEY_WIDTH + 0.7 * WHITE_KEY_WIDTH, 0, BLACK_KEY_WIDTH, BLACK_KEY_HEIGHT))
            x = i * WHITE_KEY_WIDTH + 0.7 * WHITE_KEY_WIDTH
            if note_name in NOTE_TO_KEYBOARD:
                text_surface = FONT.render(NOTE_TO_KEYBOARD[note_name], True, WHITE)
                text_rect = text_surface.get_rect(center=(x + BLACK_KEY_WIDTH/2, BLACK_KEY_HEIGHT - 15))
                SCREEN.blit(text_surface, text_rect)

    pygame.display.flip()
    
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            key_name = pygame.key.name(event.key)
            if key_name in KEYBOARD_MAP and KEYBOARD_MAP[key_name] not in notes_pressed:
                note = KEYBOARD_MAP[key_name]
                notes_pressed[note] = current_time
                NOTE_CHANNELS[note].play(NOTE_SOUNDS[note], loops=-1)
        elif event.type == pygame.KEYUP:
            key_name = pygame.key.name(event.key)
            if key_name in KEYBOARD_MAP:
                note = KEYBOARD_MAP[key_name]
                if note in notes_pressed:
                    start_time = notes_pressed.pop(note)
                    notes_played.append((note, start_time, current_time))
                    NOTE_CHANNELS[note].stop()
    
    pygame.time.delay(10)

pygame.quit()
