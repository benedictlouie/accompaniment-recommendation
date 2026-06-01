import pygame
import time
import threading
import queue
from transcribe.transcriber import Transcriber
from transcribe.transcriber_visualiser import TranscriberVisualiser
from engines.factory import create_engine
from accompaniment.accompaniment_system import AccompanimentSystem
from utils.constants import FONT_BIG, FONT_MED, FONT_SMALL, BLACK, WHITE, GRAY, BLUE, GREEN, SAMPLE_RATE, BEATS_PER_BAR
from utils.metronome import Metronome

# ==========================================================
# PYGAME INIT
# ==========================================================

pygame.init()
pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2)

WIDTH, HEIGHT = 1200, 700
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))

pygame.display.set_caption("Real-Time Harmoniser")

clock = pygame.time.Clock()


# ==========================================================
# AUDIO / ENGINE SETUP
# ==========================================================

transcriber = Transcriber()
transcriber.start()

tempo = 100

metronome = Metronome(tempo, WIDTH)

engine = create_engine("transformer", tempo)

accompaniment_system = AccompanimentSystem()

visualiser = TranscriberVisualiser(
    transcriber,
    WIDTH,
    HEIGHT
)

pygame.mixer.set_num_channels(8)


# ==========================================================
# BACKGROUND BEAT PROCESSING
# ==========================================================

beat_queue = queue.Queue()

predicted_chord = "-"


def beat_worker():

    global predicted_chord

    while True:

        item = beat_queue.get()

        if item is None:
            break

        processed_notes, prev_beat_start, prev_beat, current_beat, beat_fire_time = item

        chord, duration = engine.process_beat(
            processed_notes,
            prev_beat_start,
            prev_beat,
        )
        print(f"[CHORD] beat={prev_beat}  chord={chord}  latency={1000*(time.time()-beat_fire_time):.0f}ms")

        melody = engine.last_bar

        accompaniment_system.play_beat(
            melody,
            chord,
            tempo,
            current_beat,
        )

        if chord:
            predicted_chord = chord


threading.Thread(
    target=beat_worker,
    daemon=True
).start()


# ==========================================================
# MAIN LOOP STATE
# ==========================================================

running = True

current_beat = 1
last_beat_time = time.time()

last_note = "-"


# ==========================================================
# MAIN LOOP
# ==========================================================

while running:

    now = time.time()

    # ----------------------------------
    # capture audio every frame (16th)
    # ----------------------------------

    note = transcriber.capture_16th()

    if note not in ["quiet", "no pitch"]:
        last_note = note

    beat_happened = False

    # ----------------------------------
    # beat timing
    # ----------------------------------

    if now - last_beat_time >= engine.beat_duration:

        beat_happened = True

        beat_start = last_beat_time
        last_beat_time += engine.beat_duration

        accurate_notes = transcriber.capture_beat_4_16ths()

        # The buffer always contains audio from the PREVIOUS beat's window.
        # Use that beat's start time and beat index so the engine's strong_flag
        # and history labelling are correct.
        prev_beat_start = beat_start - engine.beat_duration
        prev_beat = (current_beat - 2) % BEATS_PER_BAR + 1

        processed_notes = []

        for i, note in enumerate(accurate_notes):

            if note not in ["quiet", "no pitch"]:

                slice_start = prev_beat_start + i * engine.step_duration
                slice_end = slice_start + engine.step_duration

                processed_notes.append(
                    (note, slice_start, slice_end)
                )

        # send heavy work to background thread
        # prev_beat_start / prev_beat: tell the engine which beat's audio this is
        # current_beat: tells play_beat which loop step to play right now
        print(f"[BEAT] beat={current_beat}")
        beat_queue.put(
            (processed_notes, prev_beat_start, prev_beat, current_beat, now)
        )

        metronome.mute(predicted_chord not in ['-', 'N'])
        metronome.click()

        metronome.advance()
        current_beat = metronome.current_beat

    # ----------------------------------
    # update visualiser
    # ----------------------------------

    visualiser.update(
        beat_happened,
        current_beat
    )

    # ----------------------------------
    # DRAW UI
    # ----------------------------------

    SCREEN.fill(BLACK)

    bpm_text = FONT_BIG.render(
        f"{tempo} BPM",
        True,
        WHITE
    )

    SCREEN.blit(
        bpm_text,
        (WIDTH // 2 - 90, 30)
    )

    metronome.draw(SCREEN)

    SCREEN.blit(
        FONT_SMALL.render(
            "Detected Note",
            True,
            GRAY,
        ),
        (80, 150),
    )

    SCREEN.blit(
        FONT_MED.render(
            last_note,
            True,
            GREEN,
        ),
        (80, 180),
    )

    SCREEN.blit(
        FONT_SMALL.render(
            "Predicted Harmony",
            True,
            GRAY,
        ),
        (WIDTH * 0.8, 150),
    )

    SCREEN.blit(
        FONT_MED.render(
            predicted_chord,
            True,
            BLUE,
        ),
        (WIDTH * 0.8, 180),
    )

    # ----------------------------------
    # input level meter
    # ----------------------------------

    level = min(transcriber.current_amplitude * 600, 150)

    pygame.draw.rect(
        SCREEN,
        GREEN,
        (80, 100, float(level), 20),
    )

    pygame.draw.rect(
        SCREEN,
        WHITE,
        (80, 100, 150, 20),
        2,
    )

    # ----------------------------------
    # draw pitch visualisation
    # ----------------------------------

    visualiser.draw(SCREEN)

    # ----------------------------------
    # events
    # ----------------------------------

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False
        if metronome.handle_event(event):
            tempo = metronome.tempo
            engine.set_tempo(tempo)

    pygame.display.flip()

    clock.tick(60)


# ==========================================================
# CLEANUP
# ==========================================================

beat_queue.put(None)

transcriber.stop()

pygame.quit()