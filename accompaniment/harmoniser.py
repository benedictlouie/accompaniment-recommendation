import pygame
import numpy as np
import time
from transcribe.transcriber import Transcriber
from engines.factory import create_engine
from utils.constants import FONT_BIG, FONT_MED, FONT_SMALL, BLACK, WHITE, GRAY, BLUE, RED, GREEN, SAMPLE_RATE
from accompaniment.accompaniment_system import AccompanimentSystem
from utils.metronome import Metronome

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
# TEMPO & METRONOME
# =====================================================
tempo = 100
current_beat = 1
metronome = Metronome(tempo, WIDTH)

# Beat & sixteenth note clocks
last_beat_time = time.time()
last_sixteenth_time = time.time()

# =====================================================
# ENGINE
# =====================================================
engine = create_engine("transformer", tempo)
accompaniment_system = AccompanimentSystem()

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

        # Get accurate 4x16ths from full beat audio
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

        # Clear old memory
        note_memory.clear()

        # ===========================
        # Metronome click with mute logic
        # ===========================
        metronome.mute(chord != "N")  # <-- mute if no chord
        metronome.click()

        # Play harmony
        melody = engine.last_bar
        accompaniment_system.play_beat(melody, chord, tempo, current_beat)
        if chord:
            predicted_chord = chord

        # Advance beat
        metronome.advance()
        current_beat = metronome.current_beat

    # ===========================
    # DRAW
    # ===========================
    SCREEN.fill(BLACK)

    # BPM
    bpm_text = FONT_BIG.render(f"{tempo} BPM", True, WHITE)
    SCREEN.blit(bpm_text, (WIDTH//2 - 90, 30))

    # Beat dots
    metronome.draw(SCREEN)

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

    pygame.display.flip()

    # ===========================
    # EVENTS
    # ===========================
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # BPM button clicks
        if event.type == pygame.MOUSEBUTTONDOWN:
            if plus_button_rect.collidepoint(event.pos):
                tempo += 5
                engine.set_tempo(tempo)
                metronome.set_tempo(tempo)
            elif minus_button_rect.collidepoint(event.pos):
                tempo = max(20, tempo - 5)
                engine.set_tempo(tempo)
                metronome.set_tempo(tempo)

    pygame.time.delay(10)

transcriber.stop()
pygame.quit()