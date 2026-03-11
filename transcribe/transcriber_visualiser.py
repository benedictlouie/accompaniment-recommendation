import time
import pygame
import numpy as np
from transcribe.transcriber import Transcriber
from utils.metronome import Metronome
from utils.constants import STEPS_PER_BEAT, BEATS_PER_BAR, FONT_BIG, FONT_SMALL, BLACK, DARK_GRAY, WHITE, GRAY, GREEN, RED


class TranscriberVisualiser:

    def __init__(self, transcriber, width, height):

        self.transcriber = transcriber
        self.WIDTH = width
        self.HEIGHT = height

        # =============================
        # SPECTROGRAM SETTINGS
        # =============================

        self.SPEC_X = 120
        self.SPEC_Y = 430
        self.SPEC_W = width - 180
        self.SPEC_H = 250

        # =============================
        # MIDI GRID
        # =============================

        self.MIDI_MIN = 36
        self.MIDI_MAX = 96
        self.MIDI_RANGE = self.MIDI_MAX - self.MIDI_MIN

        # =============================
        # BEAT LINES
        # =============================

        self.beat_line_positions = []

    # ==========================================================
    # UPDATE
    # ==========================================================

    def update(self, beat_happened, current_beat):

        col_width = self.SPEC_W / self.transcriber.SPEC_WIDTH
        shift_amount = col_width * STEPS_PER_BEAT

        if beat_happened:

            self.beat_line_positions = [
                x - shift_amount
                for x in self.beat_line_positions
                if x >= self.SPEC_X
            ]

            if current_beat % BEATS_PER_BAR == 1:
                self.beat_line_positions.append(
                    self.SPEC_X + self.SPEC_W
                )

    # ==========================================================
    # DRAW
    # ==========================================================

    def draw(self, screen):

        self.draw_current_note(screen)
        self.draw_waveform(screen)
        self.draw_spectrogram(screen)
        self.draw_midi_grid(screen)
        self.draw_current_pitch(screen)
        self.draw_beat_lines(screen)

    # ==========================================================
    # CURRENT NOTE
    # ==========================================================

    def draw_current_note(self, screen):

        if (
            self.transcriber.current_pitch
            and self.transcriber.current_amplitude
            > self.transcriber.VOLUME_THRESHOLD
        ):

            midi = int(
                self.transcriber.hz_to_midi(
                    self.transcriber.current_pitch
                )
            )

            note_name = self.transcriber.midi_to_note_name(midi)

            text = FONT_BIG.render(note_name, True, WHITE)

            screen.blit(
                text,
                (
                    self.WIDTH // 2 - text.get_width() // 2,
                    130,
                ),
            )

    # ==========================================================
    # WAVEFORM
    # ==========================================================

    def draw_waveform(self, screen):

        if len(self.transcriber.waveform_buffer) <= 1:
            return

        data = np.array(self.transcriber.waveform_buffer)
        data = data / (np.max(np.abs(data)) + 1e-6)

        mid = 300
        scale_x = (self.WIDTH - 60) / len(data)
        scale_y = 70

        points = [
            (30 + i * scale_x, mid - sample * scale_y)
            for i, sample in enumerate(data)
        ]

        pygame.draw.lines(screen, GREEN, False, points, 2)

    # ==========================================================
    # SPECTROGRAM
    # ==========================================================

    def draw_spectrogram(self, screen):

        pygame.draw.rect(
            screen,
            DARK_GRAY,
            (self.SPEC_X, self.SPEC_Y, self.SPEC_W, self.SPEC_H),
        )

        col_width = self.SPEC_W / self.transcriber.SPEC_WIDTH
        row_height = self.SPEC_H / self.transcriber.SPEC_HEIGHT

        max_val = np.max(self.transcriber.spectrogram) + 1e-6

        for x in range(self.transcriber.SPEC_WIDTH):

            for y in range(self.transcriber.SPEC_HEIGHT):

                intensity = (
                    self.transcriber.spectrogram[y, x] / max_val
                )

                if intensity < 0.03:
                    continue

                color = [i * intensity for i in RED]

                relative = y / self.transcriber.SPEC_HEIGHT

                draw_y = self.SPEC_Y + (
                    relative + (1 / self.transcriber.SPEC_HEIGHT)
                ) * self.SPEC_H

                rect = pygame.Rect(
                    self.SPEC_X + x * col_width,
                    draw_y,
                    col_width,
                    row_height,
                )

                pygame.draw.rect(screen, color, rect)

    # ==========================================================
    # MIDI GRID
    # ==========================================================

    def draw_midi_grid(self, screen):

        for midi in range(self.MIDI_MIN, self.MIDI_MAX + 1):

            relative = (midi - self.MIDI_MIN) / self.MIDI_RANGE

            y_pos = self.SPEC_Y + self.SPEC_H - (relative * self.SPEC_H)

            if midi % 12 == 0:

                pygame.draw.line(
                    screen,
                    GRAY,
                    (self.SPEC_X - 5, y_pos),
                    (self.SPEC_X + self.SPEC_W, y_pos),
                    2,
                )

                label = FONT_SMALL.render(
                    self.transcriber.midi_to_note_name(midi),
                    True,
                    WHITE,
                )

                screen.blit(label, (self.SPEC_X - 60, y_pos - 8))

            else:

                pygame.draw.line(
                    screen,
                    DARK_GRAY,
                    (self.SPEC_X, y_pos),
                    (self.SPEC_X + self.SPEC_W, y_pos),
                    1,
                )

    # ==========================================================
    # CURRENT PITCH LINE
    # ==========================================================

    def draw_current_pitch(self, screen):

        if (
            not self.transcriber.current_pitch
            or self.transcriber.current_amplitude
            < self.transcriber.VOLUME_THRESHOLD
        ):
            return

        midi = int(
            self.transcriber.hz_to_midi(
                self.transcriber.current_pitch
            )
        )

        if not (self.MIDI_MIN <= midi <= self.MIDI_MAX):
            return

        relative = (midi - self.MIDI_MIN) / self.MIDI_RANGE

        y_pos = self.SPEC_Y + self.SPEC_H - (relative * self.SPEC_H)

        pygame.draw.line(
            screen,
            GREEN,
            (self.SPEC_X, y_pos),
            (self.SPEC_X + self.SPEC_W, y_pos),
            3,
        )

    # ==========================================================
    # BEAT LINES
    # ==========================================================

    def draw_beat_lines(self, screen):

        for x in self.beat_line_positions:

            pygame.draw.rect(
                screen,
                GRAY,
                (x, self.SPEC_Y, 1, self.SPEC_H),
            )


# ==============================================================
# STANDALONE TEST LOOP (your original visualiser)
# ==============================================================

if __name__ == "__main__":

    pygame.init()

    WIDTH, HEIGHT = 1200, 720
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    pygame.display.set_caption("16th Note Quantised Pitch Visualiser")

    clock = pygame.time.Clock()

    BPM = 100

    transcriber = Transcriber()
    transcriber.start()

    metronome = Metronome(BPM, WIDTH)

    visualiser = TranscriberVisualiser(
        transcriber,
        WIDTH,
        HEIGHT
    )

    running = True

    while running:

        screen.fill(BLACK)

        beat_happened, beat_start_time = metronome.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if beat_happened:

            metronome.click()
            metronome.advance()

            for _ in range(STEPS_PER_BEAT):
                transcriber.capture_16th()

            accurate_results = transcriber.capture_beat_4_16ths()

            print(
                f"Beat {metronome.current_beat}:",
                accurate_results
            )

        visualiser.update(
            beat_happened,
            metronome.current_beat
        )

        pygame.draw.rect(
            screen,
            DARK_GRAY,
            (20, 20, WIDTH - 40, 170),
            border_radius=12,
        )

        pygame.draw.rect(
            screen,
            DARK_GRAY,
            (20, 210, WIDTH - 40, 180),
            border_radius=12,
        )

        pygame.draw.rect(
            screen,
            DARK_GRAY,
            (20, 410, WIDTH - 40, 280),
            border_radius=12,
        )

        metronome.draw(screen, 80)

        visualiser.draw(screen)

        pygame.display.flip()
        clock.tick(60)

    transcriber.stop()
    pygame.quit()