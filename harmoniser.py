import pygame
import numpy as np
import time
from transcribe.transcriber import Transcriber
from engines.factory import create_engine
from utils.constants import *
from utils.accompaniment import play_harmony

# =====================================================
# PYGAME INIT
# =====================================================

pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=2)

WIDTH, HEIGHT = 1000, 500
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Real-Time Harmoniser")

FONT_BIG = pygame.font.SysFont("Arial", 42)
FONT_MED = pygame.font.SysFont("Arial", 28)
FONT_SMALL = pygame.font.SysFont("Arial", 18)

BG = (25, 27, 32)
WHITE = (240, 240, 240)
GRAY = (120, 120, 120)
BLUE = (90, 170, 255)
RED = (255, 90, 90)
GREEN = (80, 220, 140)

# =====================================================
# ENGINE
# =====================================================

engine = create_engine("transformer")

# =====================================================
# TRANSCRIBER
# =====================================================

transcriber = Transcriber()
transcriber.start()

# =====================================================
# TEMPO
# =====================================================

tempo = 100
BEATS_PER_BAR = 4
current_beat = 1

# Beat & sixteenth note clocks
last_beat_time = time.time()
last_sixteenth_time = time.time()

def beat_duration():
    return 60.0 / tempo

def sixteenth_duration():
    return beat_duration() / 4

# =====================================================
# AUDIO OUTPUT
# =====================================================

pygame.mixer.set_num_channels(8)
channels = [pygame.mixer.Channel(i) for i in range(4)]

# =====================================================
# MAIN LOOP
# =====================================================

running = True
predicted_chord = "-"
last_note = "-"
note_memory = []

# Buttons
button_width, button_height = 50, 50
plus_button_rect = pygame.Rect(WIDTH//2 + 100, 25, button_width, button_height)
minus_button_rect = pygame.Rect(WIDTH//2 - 150, 25, button_width, button_height)

while running:

    now = time.time()

    # ===========================
    # Sixteenth note clock (4x per beat)
    # ===========================
    if now - last_sixteenth_time >= sixteenth_duration():
        last_sixteenth_time += sixteenth_duration()

        note = transcriber.capture_16th()

        if note not in ["quiet", "no pitch"]:
            last_note = note
            note_memory.append((note, last_sixteenth_time, now))

    # ===========================
    # Beat clock (once per beat)
    # ===========================
    if now - last_beat_time >= beat_duration():
        beat_start = last_beat_time
        last_beat_time += beat_duration()

        # Generate harmony for the beat using the notes captured in this beat
        chord, duration = engine.process_beat(
            note_memory,
            beat_start,
            current_beat
        )

        note_memory.clear()

        # Metronome click
        if current_beat % BEATS_PER_BAR == 0:
            CLICK_SOUND_STRONG.play()
        else:
            CLICK_SOUND.play()

        # Play harmony
        if chord:
            predicted_chord = chord
            play_harmony(chord, duration, channels)

        current_beat += 1
        if current_beat > BEATS_PER_BAR:
            current_beat = 1

    # ===========================
    # DRAW
    # ===========================
    SCREEN.fill(BG)

    # BPM
    bpm_text = FONT_BIG.render(f"{tempo} BPM", True, WHITE)
    SCREEN.blit(bpm_text, (WIDTH//2 - 90, 30))

    # Beat dots
    for i in range(BEATS_PER_BAR):
        color = RED if (i+1)==current_beat else GRAY
        pygame.draw.circle(SCREEN, color, (400 + i*60, 100), 12)

    # Detected note
    SCREEN.blit(FONT_SMALL.render("Detected Note", True, GRAY), (80, 200))
    SCREEN.blit(FONT_MED.render(last_note, True, GREEN), (80, 230))

    # Chord
    SCREEN.blit(FONT_SMALL.render("Predicted Harmony", True, GRAY), (650, 200))
    SCREEN.blit(FONT_MED.render(predicted_chord, True, BLUE), (650, 230))

    # Mic level meter
    level = min(transcriber.current_amplitude * 150, 150)
    pygame.draw.rect(SCREEN, GREEN, (80, 150, float(level), 20))
    pygame.draw.rect(SCREEN, WHITE, (80, 150, 150, 20), 2)

    # ===========================
    # BPM BUTTONS
    # ===========================
    CENTER_X = WIDTH // 2
    BTN_MINUS = pygame.Rect(CENTER_X - 160, 35, 40, 40)
    BTN_PLUS  = pygame.Rect(CENTER_X + 120, 35, 40, 40)
    pygame.draw.rect(SCREEN, BLUE, BTN_MINUS, border_radius=8)
    pygame.draw.rect(SCREEN, BLUE, BTN_PLUS, border_radius=8)
    SCREEN.blit(FONT_BIG.render("-", True, WHITE),
                (BTN_MINUS.centerx - 8, BTN_MINUS.centery - 20))
    SCREEN.blit(FONT_BIG.render("+", True, WHITE),
                (BTN_PLUS.centerx - 10, BTN_PLUS.centery - 20))

    pygame.display.flip()

    # ===========================
    # EVENTS
    # ===========================
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # Check for button clicks
        if event.type == pygame.MOUSEBUTTONDOWN:
            if plus_button_rect.collidepoint(event.pos):
                tempo += 5
            elif minus_button_rect.collidepoint(event.pos):
                tempo = max(20, tempo - 5)

    pygame.time.delay(10)

transcriber.stop()
pygame.quit()