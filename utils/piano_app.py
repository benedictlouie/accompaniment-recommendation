import pygame
import numpy as np
import time
from CRF.chord_engine import ChordEngine
from utils.constants import *

pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=2)

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

WIDTH, HEIGHT = 1100, 380
TOP_BAR_HEIGHT = 100
PIANO_TOP = TOP_BAR_HEIGHT
PIANO_HEIGHT = HEIGHT - TOP_BAR_HEIGHT

SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Real-time Piano Accompaniment")

FONT_SMALL = pygame.font.SysFont("Arial", 16)
FONT_MED = pygame.font.SysFont("Arial", 22)
FONT_BIG = pygame.font.SysFont("Arial", 40)

# Colors
BG = (28, 30, 34)
TOP_BG = (40, 44, 52)
WHITE = (240, 240, 240)
BLACK = (20, 20, 20)
BLUE = (80, 160, 255)
RED = (255, 80, 80)
GREEN = (120, 220, 120)
GRAY = (120, 120, 120)

# --------------------------------------------------
# SOUND
# --------------------------------------------------

def generate_note_sound(freq, duration=1.0):
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = 0.3 * np.sin(2 * np.pi * freq * t)
    wave = np.int16(wave * 32767)
    stereo = np.column_stack([wave, wave])
    return pygame.sndarray.make_sound(stereo)

NOTE_SOUNDS = {
    note: generate_note_sound(freq * 2)
    for note, freq in NOTE_FREQS.items()
}

pygame.mixer.set_num_channels(len(NOTE_FREQS) + 8)

NOTE_CHANNELS = {
    note: pygame.mixer.Channel(i)
    for i, note in enumerate(NOTE_FREQS)
}

HARMONY_CHANNELS = [
    pygame.mixer.Channel(len(NOTE_FREQS) + i)
    for i in range(4)
]

# --------------------------------------------------
# TEMPO
# --------------------------------------------------

TEMPO_OPTIONS = list(range(40, 241, 5))
tempo_index = TEMPO_OPTIONS.index(120)
tempo = TEMPO_OPTIONS[tempo_index]

CENTER_X = WIDTH // 2
BTN_MINUS = pygame.Rect(CENTER_X - 160, 35, 40, 40)
BTN_PLUS  = pygame.Rect(CENTER_X + 120, 35, 40, 40)


def beat_duration():
    return 60.0 / tempo

# --------------------------------------------------
# METRONOME
# --------------------------------------------------

BEATS_PER_BAR = 4
current_beat = 1
last_beat_time = time.time()
bar_start_time = last_beat_time

# --------------------------------------------------
# PIANO LAYOUT
# --------------------------------------------------

WHITE_KEYS = ['C', 'D', 'E', 'F', 'G', 'A', 'B'] * 2
BLACK_KEYS = ['C#', 'D#', '', 'F#', 'G#', 'A#', '',
              'C#', 'D#', '', 'F#', 'G#', 'A#', '']

WHITE_KEY_WIDTH = WIDTH // len(WHITE_KEYS)
WHITE_KEY_HEIGHT = PIANO_HEIGHT
BLACK_KEY_WIDTH = WHITE_KEY_WIDTH // 2
BLACK_KEY_HEIGHT = int(PIANO_HEIGHT * 0.6)

# --------------------------------------------------
# NOTE TRACKING
# --------------------------------------------------

notes_pressed = {}
notes_played = []

engine = ChordEngine()

predicted_chord_display = "-"

# --------------------------------------------------
# PLAY HARMONY
# --------------------------------------------------

def play_harmony(chord_name, duration):
    if chord_name not in CHORD_TO_TETRAD:
        return

    for i, midi in enumerate(CHORD_TO_TETRAD[chord_name]):
        freq = 440 * 2 ** (1 + (midi - 69) / 12)
        sound = generate_note_sound(freq, duration)
        HARMONY_CHANNELS[i].play(sound)

# --------------------------------------------------
# MAIN LOOP
# --------------------------------------------------

running = True

while running:

    current_time = time.time()

    # ------------------ METRONOME ------------------

    if current_time - last_beat_time >= beat_duration():
        last_beat_time = current_time

        if current_beat < BEATS_PER_BAR:
            CLICK_SOUND.play()
            current_beat += 1
        else:
            CLICK_SOUND_STRONG.play()

            bar_end_time = current_time
            bar_proportions = {}

            for note, start, end in notes_played:
                start = max(start, bar_start_time)
                end = min(end, bar_end_time)
                duration = max(0, end - start)
                bar_proportions[note] = bar_proportions.get(note, 0) + duration

            for note in bar_proportions:
                bar_proportions[note] /= (beat_duration() * BEATS_PER_BAR)

            chord = engine.process_bar(bar_proportions)

            if chord:
                play_harmony(chord, beat_duration() * BEATS_PER_BAR)
                predicted_chord_display = chord

            notes_played.clear()
            current_beat = 1
            bar_start_time = current_time

    # ------------------ DRAW ------------------

    SCREEN.fill(BG)

    # TOP BAR
    pygame.draw.rect(SCREEN, TOP_BG, (0, 0, WIDTH, TOP_BAR_HEIGHT))

    # BPM display
    bpm_text = FONT_BIG.render(f"{tempo} BPM", True, WHITE)
    SCREEN.blit(bpm_text, (WIDTH // 2 - 80, 30))

    # Tempo buttons
    pygame.draw.rect(SCREEN, BLUE, BTN_MINUS, border_radius=8)
    pygame.draw.rect(SCREEN, BLUE, BTN_PLUS, border_radius=8)

    SCREEN.blit(FONT_BIG.render("-", True, WHITE),
                (BTN_MINUS.centerx - 8, BTN_MINUS.centery - 20))

    SCREEN.blit(FONT_BIG.render("+", True, WHITE),
                (BTN_PLUS.centerx - 10, BTN_PLUS.centery - 20))

    # Beat indicator (circles)
    DOT_SPACING = 50
    TOTAL_WIDTH = DOT_SPACING * 3
    START_X = WIDTH // 2 - TOTAL_WIDTH // 2

    for i in range(4):
        x = START_X + i * DOT_SPACING
        y = 20
        color = RED if (i + 1) == current_beat else GRAY
        pygame.draw.circle(SCREEN, color, (x, y), 10)

    # Chord display
    chord_label = FONT_SMALL.render("Predicted Chord", True, GRAY)
    SCREEN.blit(chord_label, (WIDTH - 250, 25))

    chord_text = FONT_MED.render(predicted_chord_display, True, BLUE)
    SCREEN.blit(chord_text, (WIDTH - 250, 50))

    # ------------------ DRAW PIANO ------------------

    for i, key in enumerate(WHITE_KEYS):
        note_name = key + ('4' if i < 7 else '5')
        rect = pygame.Rect(i * WHITE_KEY_WIDTH,
                           PIANO_TOP,
                           WHITE_KEY_WIDTH,
                           WHITE_KEY_HEIGHT)

        color = BLUE if note_name in notes_pressed else WHITE
        pygame.draw.rect(SCREEN, color, rect)
        pygame.draw.rect(SCREEN, BLACK, rect, 2)

        if note_name in NOTE_TO_KEYBOARD:
            label = FONT_SMALL.render(NOTE_TO_KEYBOARD[note_name], True, BLACK)
            SCREEN.blit(label, (rect.centerx - 6, rect.bottom - 20))

    for i, key in enumerate(BLACK_KEYS):
        if key:
            note_name = key + ('4' if i < 7 else '5')
            rect = pygame.Rect(
                i * WHITE_KEY_WIDTH + 0.7 * WHITE_KEY_WIDTH,
                PIANO_TOP,
                BLACK_KEY_WIDTH,
                BLACK_KEY_HEIGHT
            )

            color = BLUE if note_name in notes_pressed else BLACK
            pygame.draw.rect(SCREEN, color, rect)

            if note_name in NOTE_TO_KEYBOARD:
                label = FONT_SMALL.render(NOTE_TO_KEYBOARD[note_name], True, WHITE)
                SCREEN.blit(label, (rect.centerx - 6, rect.bottom - 18))

    pygame.display.flip()

    # ------------------ EVENTS ------------------

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if BTN_MINUS.collidepoint(event.pos):
                tempo_index = max(0, tempo_index - 1)
                tempo = TEMPO_OPTIONS[tempo_index]

            if BTN_PLUS.collidepoint(event.pos):
                tempo_index = min(len(TEMPO_OPTIONS) - 1, tempo_index + 1)
                tempo = TEMPO_OPTIONS[tempo_index]

        elif event.type == pygame.KEYDOWN:
            key_name = pygame.key.name(event.key)

            if key_name in KEYBOARD_MAP:
                note = KEYBOARD_MAP[key_name]
                if note not in notes_pressed:
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
