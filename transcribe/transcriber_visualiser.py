import time
import pygame
import numpy as np
from transcribe.transcriber import Transcriber

# =============================
# METRONOME SETTINGS
# =============================
BPM = 100
SUBDIV = 4

def update_timing():
    global SECONDS_PER_BEAT, SUBDIV_DURATION
    SECONDS_PER_BEAT = 60 / BPM
    SUBDIV_DURATION = SECONDS_PER_BEAT / SUBDIV

update_timing()

# =============================
# PYGAME INIT
# =============================
pygame.init()
WIDTH, HEIGHT = 1200, 720
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("16th Note Quantised Pitch Visualiser")

clock = pygame.time.Clock()
pygame.mixer.init()
click = pygame.mixer.Sound("utils/click.wav")

# =============================
# FONTS
# =============================
font_big = pygame.font.SysFont("Arial", 64)
font_medium = pygame.font.SysFont("Arial", 28)
font_small = pygame.font.SysFont("Arial", 18)

# =============================
# COLORS
# =============================
BG = (18, 18, 18)
PANEL = (28, 28, 28)
WHITE = (235, 235, 235)
GREY = (140, 140, 140)
GRID = (55, 55, 55)
OCTAVE = (95, 95, 95)
GREEN = (0, 255, 140)
ACCENT = (255, 140, 0)

# =============================
# BUTTON CLASS
# =============================
class Button:
    def __init__(self, x, y, w, h, text):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text

    def draw(self, surface):
        pygame.draw.rect(surface, PANEL, self.rect, border_radius=8)
        pygame.draw.rect(surface, ACCENT, self.rect, 2, border_radius=8)
        label = font_medium.render(self.text, True, WHITE)
        surface.blit(label, (
            self.rect.centerx - label.get_width() // 2,
            self.rect.centery - label.get_height() // 2
        ))

    def clicked(self, pos):
        return self.rect.collidepoint(pos)

# =============================
# CREATE BUTTONS
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
MIDI_MAX = 84
MIDI_RANGE = MIDI_MAX - MIDI_MIN

# =============================
# MAIN LOOP
# =============================
start_time = time.time()
subdiv_index = 0
beat_count = 0
beat_results = []
running = True

while running:
    screen.fill(BG)
    now = time.time()

    # ================= EVENTS =================
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.MOUSEBUTTONDOWN:
            if btn_plus.clicked(event.pos):
                BPM += 5
                update_timing()
            if btn_minus.clicked(event.pos):
                BPM = max(20, BPM - 5)
                update_timing()

    # ================= METRONOME =================
    next_tick = start_time + subdiv_index * SUBDIV_DURATION

    if now >= next_tick:
        if subdiv_index % SUBDIV == 0:
            click.play()

        result = transcriber.capture_16th()
        beat_results.append(result)

        subdiv_index += 1

        if len(beat_results) == SUBDIV:
            print(f"Beat {beat_count + 1}: {beat_results}")
            beat_results = []
            beat_count += 1

    # ================= PANELS =================
    pygame.draw.rect(screen, PANEL, (20, 20, WIDTH - 40, 170), border_radius=12)
    pygame.draw.rect(screen, PANEL, (20, 210, WIDTH - 40, 180), border_radius=12)
    pygame.draw.rect(screen, PANEL, (20, 410, WIDTH - 40, 280), border_radius=12)

    # ================= CURRENT NOTE =================
    current_midi = None

    if (transcriber.current_pitch and
        transcriber.current_amplitude > transcriber.VOLUME_THRESHOLD):

        current_midi = int(transcriber.hz_to_midi(transcriber.current_pitch))
        note_name = transcriber.midi_to_note_name(current_midi)

        text = font_big.render(note_name, True, WHITE)
        screen.blit(text, (
            WIDTH // 2 - text.get_width() // 2,
            60
        ))

    # ================= BPM =================
    bpm_text = font_medium.render(f"BPM: {BPM}", True, GREY)
    screen.blit(bpm_text, (40, 60))
    btn_minus.draw(screen)
    btn_plus.draw(screen)

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
            pygame.draw.line(screen, OCTAVE,
                             (SPEC_X-5, y_pos),
                             (SPEC_X + SPEC_W, y_pos), 2)

            label = font_small.render(
                transcriber.midi_to_note_name(midi),
                True,
                (210,210,210)
            )
            screen.blit(label, (SPEC_X-60, y_pos - 8))
        else:
            pygame.draw.line(screen, GRID,
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
