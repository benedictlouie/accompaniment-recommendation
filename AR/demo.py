import pygame
import numpy as np
import torch
import time
from AR.ar_transformer import TransformerModel
from AR.inference import generate_chords
from utils.constants import *

STEPS_PER_BEAT = 4                 # 16th notes
STEPS_PER_BAR = STEPS_PER_BEAT # * BEATS_PER_BAR
STEP_DURATION = BEAT_DURATION / STEPS_PER_BEAT

# -----------------------------
# Sound generation
# -----------------------------
def generate_note_sound(freq, duration=1.0):
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = 0.3 * np.sin(2 * np.pi * freq * t)
    wave = np.int16(wave * 32767)
    return pygame.sndarray.make_sound(np.column_stack([wave, wave]))

NOTE_SOUNDS = {note: generate_note_sound(freq * 2) for note, freq in NOTE_FREQS.items()}

pygame.mixer.set_num_channels(len(NOTE_FREQS) + 8)
NOTE_CHANNELS = {note: pygame.mixer.Channel(i) for i, note in enumerate(NOTE_FREQS)}

# -----------------------------
# Window
# -----------------------------
WIDTH, HEIGHT = 800, 200
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Piano Keyboard")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (50, 50, 50)
BLUE = (0, 0, 255)

WHITE_KEYS = ['C', 'D', 'E', 'F', 'G', 'A', 'B'] * 2
WHITE_KEY_WIDTH = WIDTH // len(WHITE_KEYS)
WHITE_KEY_HEIGHT = HEIGHT
BLACK_KEY_WIDTH = WHITE_KEY_WIDTH // 2
BLACK_KEY_HEIGHT = HEIGHT * 0.6
BLACK_KEYS = ['C#', 'D#', '', 'F#', 'G#', 'A#', '',
              'C#', 'D#', '', 'F#', 'G#', 'A#', '']

# -----------------------------
# Note tracking
# -----------------------------
notes_pressed = {}          # note -> start_time
notes_played = []           # (note, start, end)

# -----------------------------
# Bar → MIDI grid
# -----------------------------
def bar_to_midi_grid(notes_played, bar_start, strong_beat: int):
    """
    Returns (BEATS_PER_BAR, 4) MIDI grid (16th notes).
    Each cell = avg MIDI pitch active during that slice, -1 if none.
    """
    grid = - np.ones(STEPS_PER_BAR)

    for step in range(STEPS_PER_BAR):
        step_start = bar_start + step * STEP_DURATION
        step_end = step_start + STEP_DURATION

        active_midis = []

        for note, start, end in notes_played:
            overlap_start = max(start, step_start)
            overlap_end = min(end, step_end)
            if overlap_end > overlap_start:
                active_midis.append(NOTE_TO_MIDI[note])

        if active_midis:
            grid[step] = np.max(active_midis)
    
    grid = np.concatenate(([strong_beat], grid))
    grid = grid.reshape(1, INPUT_DIM)
    return grid

# -----------------------------
# Model stubs
# -----------------------------
def next_chord(bar_history):
    model = TransformerModel(INPUT_DIM, NUM_CLASSES+1).to(DEVICE)
    checkpoint = torch.load("checkpoints/transformer_model.pth", map_location=DEVICE)
    model.load_state_dict(checkpoint)
    predicted_chord = generate_chords(model, bar_history[np.newaxis, :, :])[-1, -1]
    predicted_chord = CHORD_CLASSES[predicted_chord]
    return predicted_chord

def play_chord(midi_grid, bar_history):
    """
    midi_grid shape: (BEATS_PER_BAR, 4)
    """
    padding = np.tile(bar_to_midi_grid([], 0, -1), MEMORY).reshape(MEMORY, INPUT_DIM)
    bar_history = np.vstack((padding, bar_history, midi_grid))[-MEMORY:]
    
    decision = next_chord(bar_history)
    print("Predicted chord:", decision)

    for i, midi in enumerate(CHORD_TO_TETRAD[decision]):
        freq = 440 * 2 ** (1 + (midi - 69) / 12)
        wave = generate_note_sound(freq, BAR_DURATION)
        pygame.mixer.Channel(len(NOTE_FREQS) + i).play(wave)

    return bar_history

# -----------------------------
# Metronome state
# -----------------------------
last_beat_time = time.time()
current_beat = 0
bar_start_time = last_beat_time

# -----------------------------
# Main loop
# -----------------------------
running = True
bar_history = np.zeros((0, INPUT_DIM))

while running:
    current_time = time.time()

    # Metronome tick
    if current_time - last_beat_time >= BEAT_DURATION:
        
        last_beat_time = current_time

        bar_end_time = current_time

        if current_beat == 0:
            strong_beat = -1
        else:
            strong_beat = not (1 < current_beat <= BEATS_PER_BAR)

        midi_grid = bar_to_midi_grid(
            notes_played,
            bar_start_time,
            strong_beat
        )

        bar_history = play_chord(midi_grid, bar_history)
        notes_played.clear()
        bar_start_time = current_time

        current_beat += 1

        if 1 < current_beat <= BEATS_PER_BAR:
            CLICK_SOUND.play()
        else:
            CLICK_SOUND_STRONG.play()
            current_beat = 1


    # Draw keyboard
    SCREEN.fill(GRAY)

    for i, key in enumerate(WHITE_KEYS):
        note_name = key + ('4' if i < 7 else '5')
        color = BLUE if note_name in notes_pressed else WHITE
        pygame.draw.rect(SCREEN, color,
                         (i * WHITE_KEY_WIDTH, 0, WHITE_KEY_WIDTH, WHITE_KEY_HEIGHT))
        pygame.draw.rect(SCREEN, BLACK,
                         (i * WHITE_KEY_WIDTH, 0, WHITE_KEY_WIDTH, WHITE_KEY_HEIGHT), 2)

        if note_name in NOTE_TO_KEYBOARD:
            txt = FONT.render(NOTE_TO_KEYBOARD[note_name], True, BLACK)
            SCREEN.blit(txt, txt.get_rect(
                center=(i * WHITE_KEY_WIDTH + WHITE_KEY_WIDTH / 2,
                        WHITE_KEY_HEIGHT - 20)))

    for i, key in enumerate(BLACK_KEYS):
        if key:
            note_name = key + ('4' if i < 7 else '5')
            color = BLUE if note_name in notes_pressed else BLACK
            x = i * WHITE_KEY_WIDTH + 0.7 * WHITE_KEY_WIDTH
            pygame.draw.rect(SCREEN, color,
                             (x, 0, BLACK_KEY_WIDTH, BLACK_KEY_HEIGHT))

    pygame.display.flip()

    # Events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            k = pygame.key.name(event.key)
            if k in KEYBOARD_MAP and KEYBOARD_MAP[k] not in notes_pressed:
                note = KEYBOARD_MAP[k]
                notes_pressed[note] = current_time
                NOTE_CHANNELS[note].play(NOTE_SOUNDS[note], loops=-1)

        elif event.type == pygame.KEYUP:
            k = pygame.key.name(event.key)
            if k in KEYBOARD_MAP:
                note = KEYBOARD_MAP[k]
                if note in notes_pressed:
                    start = notes_pressed.pop(note)
                    notes_played.append((note, start, current_time))
                    NOTE_CHANNELS[note].stop()

    pygame.time.delay(10)

pygame.quit()
