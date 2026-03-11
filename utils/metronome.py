import pygame
import time

from utils.constants import CLICK_SOUND, CLICK_SOUND_STRONG, BEATS_PER_BAR, FONT_BIG, WHITE, GRAY, RED, BLUE

class Metronome:

    def __init__(self, tempo, width):

        self.tempo = tempo
        self.width = width

        self.beat_duration = 60.0 / tempo
        self.current_beat = 1

        self.last_beat_time = time.time()

        # sound control
        self.muted = False

        # UI buttons
        center_x = width // 2

        self.plus_button = pygame.Rect(center_x + 120, 35, 40, 40)
        self.minus_button = pygame.Rect(center_x - 160, 35, 40, 40)

    # =====================================================
    # TEMPO
    # =====================================================

    def set_tempo(self, tempo):

        self.tempo = tempo
        self.beat_duration = 60.0 / tempo

    # =====================================================
    # MUTE CONTROL
    # =====================================================

    def mute(self, state=True):
        self.muted = state

    # =====================================================
    # CLOCK UPDATE
    # =====================================================

    def update(self):

        now = time.time()

        if now - self.last_beat_time >= self.beat_duration:

            beat_start = self.last_beat_time
            self.last_beat_time += self.beat_duration

            return True, beat_start

        return False, None

    # =====================================================
    # CLICK
    # =====================================================

    def click(self):

        if self.muted:
            return

        if self.current_beat % BEATS_PER_BAR == 0:
            CLICK_SOUND_STRONG.play()
        else:
            CLICK_SOUND.play()

    # =====================================================
    # ADVANCE BEAT
    # =====================================================

    def advance(self):

        self.current_beat += 1

        if self.current_beat > BEATS_PER_BAR:
            self.current_beat = 1

    # =====================================================
    # UI DRAW
    # =====================================================

    def draw(self, screen, y_offset=0):

        # BPM text
        bpm_text = FONT_BIG.render(f"{self.tempo} BPM", True, WHITE)
        screen.blit(bpm_text, (self.width // 2 - 90, 30))

        # Beat dots
        DOT_SPACING = 50
        TOTAL_WIDTH = DOT_SPACING * (BEATS_PER_BAR - 1)
        START_X = self.width // 2 - TOTAL_WIDTH // 2

        for i in range(BEATS_PER_BAR):

            x = START_X + i * DOT_SPACING
            y = 20 + y_offset

            color = RED if (i + 1) == self.current_beat else GRAY

            pygame.draw.circle(screen, color, (x, y), 10)

        # Buttons
        pygame.draw.rect(screen, BLUE, self.minus_button, border_radius=8)
        pygame.draw.rect(screen, BLUE, self.plus_button, border_radius=8)

        screen.blit(
            FONT_BIG.render("-", True, WHITE),
            (self.minus_button.centerx - 8, self.minus_button.centery - 20)
        )

        screen.blit(
            FONT_BIG.render("+", True, WHITE),
            (self.plus_button.centerx - 10, self.plus_button.centery - 20)
        )

    # =====================================================
    # MOUSE EVENTS
    # =====================================================

    def handle_event(self, event):

        tempo_changed = False

        if event.type == pygame.MOUSEBUTTONDOWN:

            if self.plus_button.collidepoint(event.pos):
                self.set_tempo(self.tempo + 5)
                tempo_changed = True

            elif self.minus_button.collidepoint(event.pos):
                self.set_tempo(max(20, self.tempo - 5))
                tempo_changed = True

        return tempo_changed