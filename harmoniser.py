import pygame
import numpy as np
import time
from transcribe.transcriber import Transcriber
from engines.factory import create_engine
from utils.constants import CLICK_SOUND, CLICK_SOUND_STRONG, BEATS_PER_BAR, FONT_BIG, FONT_MED, FONT_SMALL, BLACK, WHITE, GRAY, BLUE, RED, GREEN, SAMPLE_RATE
from accompaniment.accompaniment import play_harmony, get_fs

# =====================================================
# PYGAME INIT
# =====================================================

pygame.init()
pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2)

WIDTH, HEIGHT = 1000, 500
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Real-Time Harmoniser")

# =====================================================
# TRANSCRIBER
# =====================================================

transcriber = Transcriber()
transcriber.start()

# =====================================================
# TEMPO
# =====================================================

tempo = 100
current_beat = 1

# Beat & sixteenth note clocks
last_beat_time = time.time()
last_sixteenth_time = time.time()

# =====================================================
# ENGINE
# =====================================================

engine = create_engine("transformer", tempo)


# --------------------------------------------------
# FLUIDSYNTH
# --------------------------------------------------
FS = get_fs(
    guitar_sf="soundfonts/Guitar.sf2",
)

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
    if now - last_sixteenth_time >= engine.step_duration:
        last_sixteenth_time += engine.step_duration

        note = transcriber.capture_16th()

        if note not in ["quiet", "no pitch"]:
            last_note = note
            note_memory.append((note, last_sixteenth_time, now))

    # ===========================
    # Beat clock (once per beat)
    # ===========================
    if now - last_beat_time >= engine.beat_duration:
        beat_start = last_beat_time
        last_beat_time += engine.beat_duration

        # 🔥 Get accurate 4x16ths from full beat audio
        accurate_notes = transcriber.capture_beat_4_16ths()

        # Convert to engine format (note, start_time, end_time)
        processed_notes = []

        for i, note in enumerate(accurate_notes):
            if note not in ["quiet", "no pitch"]:
                slice_start = beat_start + i * engine.step_duration
                slice_end = slice_start + engine.step_duration
                processed_notes.append((note, slice_start, slice_end))

        # Generate harmony from accurate notes
        chord, duration = engine.process_beat(
            processed_notes,
            beat_start,
            current_beat
        )

        # Clear old memory (we no longer use live note_memory)
        note_memory.clear()

        # Metronome click
        if current_beat % BEATS_PER_BAR == 0:
            CLICK_SOUND_STRONG.play()
        else:
            CLICK_SOUND.play()

        # Play harmony
        if chord:
            predicted_chord = chord
            play_harmony(chord, duration, FS)

        current_beat += 1
        if current_beat > BEATS_PER_BAR:
            current_beat = 1

    # ===========================
    # DRAW
    # ===========================
    SCREEN.fill(BLACK)

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
                engine.set_tempo(tempo)
            elif minus_button_rect.collidepoint(event.pos):
                tempo = max(20, tempo - 5)
                engine.set_tempo(tempo)

    pygame.time.delay(10)

transcriber.stop()
pygame.quit()