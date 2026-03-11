# transcriber_visualiser.py

import time
import pygame
import numpy as np
from transcribe.transcriber import Transcriber
from utils.metronome import Metronome
from utils.constants import STEPS_PER_BEAT, FONT_BIG, FONT_MED, FONT_SMALL, BLACK, DARK_GRAY, WHITE, GRAY, GREEN, RED

# =============================
# PYGAME INIT
# =============================
pygame.init()
WIDTH, HEIGHT = 1200, 720
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("16th Note Quantised Pitch Visualiser")
clock = pygame.time.Clock()
pygame.mixer.init()

# =============================
# METRONOME
# =============================
BPM = 100
metronome = Metronome(BPM, WIDTH)

# =============================
# BUTTON CLASS (for UI adjustments)
# =============================
class Button:
    def __init__(self, x, y, w, h, text):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text

    def draw(self, surface):
        pygame.draw.rect(surface, DARK_GRAY, self.rect, border_radius=8)
        pygame.draw.rect(surface, RED, self.rect, 2, border_radius=8)
        label = FONT_MED.render(self.text, True, WHITE)
        surface.blit(label, (
            self.rect.centerx - label.get_width() // 2,
            self.rect.centery - label.get_height() // 2
        ))

    def clicked(self, pos):
        return self.rect.collidepoint(pos)

# =============================
# CREATE BUTTONS (OPTIONAL)
# =============================
btn_minus = Button(40, 110, 60, 40, "-")
btn_plus  = Button(110, 110, 60, 40, "+")

# =============================
# CREATE TRANSCRIBER
# =============================
transcriber = Transcriber()
transcriber.start()

# =============================
# SPECTROGRAM SETTINGS
# =============================
SPEC_X = 120
SPEC_Y = 430
SPEC_W = WIDTH - 180
SPEC_H = 250

MIDI_MIN = 36
MIDI_MAX = 96
MIDI_RANGE = MIDI_MAX - MIDI_MIN

# =============================
# MAIN LOOP
# =============================
subdiv_index = 0
beat_results = []
running = True

while running:
    screen.fill(BLACK)
    now = time.time()

    # ================= EVENTS =================
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # Handle tempo buttons in the visualiser
        if event.type == pygame.MOUSEBUTTONDOWN:
            if btn_plus.clicked(event.pos):
                metronome.set_tempo(metronome.tempo + 5)
            if btn_minus.clicked(event.pos):
                metronome.set_tempo(max(20, metronome.tempo - 5))

            # Also check metronome button clicks
            metronome.handle_event(event)

    # ================= METRONOME UPDATE =================
    beat_happened, beat_start_time = metronome.update()

    if beat_happened:
        # Play the click sound
        metronome.click()
        metronome.advance()

        # Capture 16th note subdivisions
        for _ in range(STEPS_PER_BEAT):
            result = transcriber.capture_16th()
            beat_results.append(result)

        # After one beat (4 16ths), print results
        accurate_results = transcriber.capture_beat_4_16ths()
        print(f"Beat {metronome.current_beat}: {accurate_results}")
        beat_results = []

    # ================= DRAW PANELS =================
    pygame.draw.rect(screen, DARK_GRAY, (20, 20, WIDTH - 40, 170), border_radius=12)
    pygame.draw.rect(screen, DARK_GRAY, (20, 210, WIDTH - 40, 180), border_radius=12)
    pygame.draw.rect(screen, DARK_GRAY, (20, 410, WIDTH - 40, 280), border_radius=12)

    # ================= CURRENT NOTE =================
    current_midi = None
    if (transcriber.current_pitch and
        transcriber.current_amplitude > transcriber.VOLUME_THRESHOLD):

        current_midi = int(transcriber.hz_to_midi(transcriber.current_pitch))
        note_name = transcriber.midi_to_note_name(current_midi)

        text = FONT_BIG.render(note_name, True, WHITE)
        screen.blit(text, (
            WIDTH // 2 - text.get_width() // 2,
            130
        ))

    # ================= METRONOME DRAW =================
    metronome.draw(screen, 80)

    # ================= WAVEFORM =================
    if len(transcriber.waveform_buffer) > 1:
        data = np.array(transcriber.waveform_buffer)
        data = data / (np.max(np.abs(data)) + 1e-6)

        mid = 300
        scale_x = (WIDTH - 60) / len(data)
        scale_y = 70

        points = [(30 + i * scale_x, mid - sample * scale_y)
                  for i, sample in enumerate(data)]

        pygame.draw.lines(screen, GREEN, False, points, 2)

    # ================= SPECTROGRAM =================
    pygame.draw.rect(screen, (22,22,22), (SPEC_X-70, SPEC_Y, 70, SPEC_H))
    pygame.draw.rect(screen, (40,40,40), (SPEC_X, SPEC_Y, SPEC_W, SPEC_H))

    col_width = SPEC_W / transcriber.SPEC_WIDTH
    row_height = SPEC_H / transcriber.SPEC_HEIGHT
    max_val = np.max(transcriber.spectrogram) + 1e-6

    for x in range(transcriber.SPEC_WIDTH):
        for y in range(transcriber.SPEC_HEIGHT):
            intensity = transcriber.spectrogram[y, x] / max_val
            if intensity > 0.03:
                color = (
                    int(255 * intensity),
                    int(150 * intensity),
                    int(40 * intensity)
                )
                relative = y / transcriber.SPEC_HEIGHT
                draw_y = SPEC_Y + (relative + (1/transcriber.SPEC_HEIGHT)) * SPEC_H

                rect = pygame.Rect(
                    SPEC_X + x * col_width,
                    draw_y,
                    col_width,
                    row_height
                )

                pygame.draw.rect(screen, color, rect)

    # ================= MIDI GRID =================
    for midi in range(MIDI_MIN, MIDI_MAX + 1):
        relative = (midi - MIDI_MIN) / MIDI_RANGE
        y_pos = SPEC_Y + SPEC_H - (relative * SPEC_H)

        if midi % 12 == 0:
            pygame.draw.line(screen, GRAY,
                             (SPEC_X-5, y_pos),
                             (SPEC_X + SPEC_W, y_pos), 2)

            label = FONT_SMALL.render(
                transcriber.midi_to_note_name(midi),
                True, (210,210,210)
            )
            screen.blit(label, (SPEC_X-60, y_pos - 8))
        else:
            pygame.draw.line(screen, DARK_GRAY,
                             (SPEC_X, y_pos),
                             (SPEC_X + SPEC_W, y_pos), 1)

    # ================= CURRENT NOTE HIGHLIGHT =================
    if current_midi and MIDI_MIN <= current_midi <= MIDI_MAX:
        relative = (current_midi - MIDI_MIN) / MIDI_RANGE
        y_pos = SPEC_Y + SPEC_H - (relative * SPEC_H)

        pygame.draw.line(screen, (0,255,180),
                         (SPEC_X, y_pos),
                         (SPEC_X + SPEC_W, y_pos), 3)

    pygame.display.flip()
    clock.tick(60)

# ================= CLEANUP =================
transcriber.stop()
pygame.quit()