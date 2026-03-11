import pygame
import pygame.midi
import numpy as np
import time
from engines.factory import create_engine
from utils.constants import ROOTS, NOTE_FREQS, KEYBOARD_MAP, KEYBOARD_LABELS, CLICK_SOUND, CLICK_SOUND_STRONG, BEATS_PER_BAR, SAMPLE_RATE, FONT_BIG, FONT_MED, FONT_SMALL, BLACK, DARK_GRAY, WHITE, BLACK, GRAY, BLUE, RED, GREEN, WHITE_KEYS, BLACK_KEYS
from accompaniment.accompaniment_system import AccompanimentSystem, SoundGenerator

pygame.init()
pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2)
pygame.midi.init()

# --------------------------------------------------
# WINDOW
# --------------------------------------------------

WIDTH, HEIGHT = 1100, 380
TOP_BAR_HEIGHT = 100
PIANO_TOP = TOP_BAR_HEIGHT
PIANO_HEIGHT = HEIGHT - TOP_BAR_HEIGHT

SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Real-time Piano Accompaniment")


# --------------------------------------------------
# SOUND
# --------------------------------------------------

pygame.mixer.set_num_channels(len(NOTE_FREQS) + 8)

NOTE_CHANNELS = {
    note: pygame.mixer.Channel(i)
    for i, note in enumerate(NOTE_FREQS)
}

NOTE_SOUNDS = SoundGenerator().note_sounds

def play_top_note():

    # stop all notes first
    for ch in NOTE_CHANNELS.values():
        ch.stop()

    if not notes_pressed:
        return

    # choose highest note by frequency
    highest_note = max(
        notes_pressed.keys(),
        key=lambda n: NOTE_FREQS[n]
    )

    NOTE_CHANNELS[highest_note].play(
        NOTE_SOUNDS[highest_note],
        loops=-1
    )

# --------------------------------------------------
# MIDI SETUP
# --------------------------------------------------

midi_input = None

for i in range(pygame.midi.get_count()):
    info = pygame.midi.get_device_info(i)
    if info[2] == 1:
        midi_input = pygame.midi.Input(i)
        print("Connected MIDI device:", info[1])
        break


MIDI_NOTE_NAMES = ROOTS.tolist()

def midi_to_note(midi_num):

    octave = midi_num // 12 - 1
    name = MIDI_NOTE_NAMES[midi_num % 12]

    return f"{name}{octave}"


# --------------------------------------------------
# TEMPO
# --------------------------------------------------

TEMPO_OPTIONS = list(range(40, 241, 5))
tempo_index = TEMPO_OPTIONS.index(100)
tempo = TEMPO_OPTIONS[tempo_index]

CENTER_X = WIDTH // 2

BTN_MINUS = pygame.Rect(CENTER_X - 160, 35, 40, 40)
BTN_PLUS  = pygame.Rect(CENTER_X + 120, 35, 40, 40)


# --------------------------------------------------
# ENGINE
# --------------------------------------------------

ENGINE_TYPE = "transformer"

engine = create_engine(ENGINE_TYPE, tempo)

accompaniment_system = AccompanimentSystem()


# --------------------------------------------------
# METRONOME
# --------------------------------------------------

current_beat = 1
last_beat_time = time.time()


# --------------------------------------------------
# PIANO LAYOUT
# --------------------------------------------------

WHITE_KEY_WIDTH = WIDTH // len(WHITE_KEYS)
WHITE_KEY_HEIGHT = PIANO_HEIGHT

BLACK_KEY_WIDTH = WHITE_KEY_WIDTH // 2
BLACK_KEY_HEIGHT = int(PIANO_HEIGHT * 0.6)


# --------------------------------------------------
# OCTAVE SHIFT
# --------------------------------------------------

octave_offset = 0
OCTAVE_MIN = -2
OCTAVE_MAX = 1


def shift_note_octave(note, offset):
    name = note[:-1]
    octave = int(note[-1])

    new_octave = octave + offset
    new_note = f"{name}{new_octave}"

    if new_note in NOTE_FREQS:
        return new_note
    return None


def get_visible_range():
    start_oct = 4 + octave_offset
    end_oct = start_oct + 1
    return f"C{start_oct}-B{end_oct}"

# --------------------------------------------------
# NOTE TRACKING
# --------------------------------------------------

notes_pressed = {}
notes_played = []

predicted_chord_display = "-"

# --------------------------------------------------
# MAIN LOOP
# --------------------------------------------------

running = True

while running:

    current_time = time.time()

    # ---------------- MIDI INPUT ----------------

    if midi_input and midi_input.poll():

        midi_events = midi_input.read(10)

        for event in midi_events:

            data = event[0]

            status = data[0]
            note_num = data[1]
            velocity = data[2]

            note = midi_to_note(note_num)

            if note not in NOTE_FREQS:
                continue

            if status == 144 and velocity > 0:

                if note not in notes_pressed:

                    notes_pressed[note] = current_time

                    play_top_note()

            elif status == 128 or velocity == 0:

                if note in notes_pressed:

                    start = notes_pressed.pop(note)

                    notes_played.append((note, start, current_time))

                    NOTE_CHANNELS[note].stop()
                    play_top_note()

    # ---------------- BEAT CLOCK ----------------

    if current_time - last_beat_time >= engine.beat_duration:

        beat_start_time = last_beat_time
        last_beat_time += engine.beat_duration

        chord, duration = engine.process_beat(
            notes_played,
            beat_start_time,
            current_beat
        )
        if not chord or chord == 'N':
            if current_beat % BEATS_PER_BAR == 0:
                CLICK_SOUND_STRONG.play()
            else:
                CLICK_SOUND.play()

        melody = engine.last_bar
        accompaniment_system.play_beat(melody, chord, tempo, current_beat)
        if chord:
            predicted_chord_display = chord

        notes_played.clear()

        current_beat += 1
        if current_beat > BEATS_PER_BAR:
            current_beat = 1

    # ---------------- DRAW ----------------

    SCREEN.fill(BLACK)

    pygame.draw.rect(SCREEN, DARK_GRAY, (0, 0, WIDTH, TOP_BAR_HEIGHT))

    bpm_text = FONT_BIG.render(f"{tempo} BPM", True, WHITE)
    SCREEN.blit(bpm_text, (WIDTH//2 - 80, 30))

    range_text = FONT_SMALL.render(f"Range {get_visible_range()} (z/x to change)", True, WHITE)
    SCREEN.blit(range_text, (WIDTH//2 - 110, 70))

    pygame.draw.rect(SCREEN, BLUE, BTN_MINUS, border_radius=8)
    pygame.draw.rect(SCREEN, BLUE, BTN_PLUS, border_radius=8)

    SCREEN.blit(FONT_BIG.render("-", True, WHITE),
                (BTN_MINUS.centerx - 8, BTN_MINUS.centery - 20))
    SCREEN.blit(FONT_BIG.render("+", True, WHITE),
                (BTN_PLUS.centerx - 10, BTN_PLUS.centery - 20))

    # Beat dots

    DOT_SPACING = 50
    TOTAL_WIDTH = DOT_SPACING * (BEATS_PER_BAR - 1)
    START_X = WIDTH // 2 - TOTAL_WIDTH // 2

    for i in range(BEATS_PER_BAR):
        x = START_X + i * DOT_SPACING
        y = 20
        color = RED if (i + 1) == current_beat else GRAY
        pygame.draw.circle(SCREEN, color, (x, y), 10)

    # Chord display
    chord_label = FONT_SMALL.render("Predicted Chord", True, GRAY)
    SCREEN.blit(chord_label, (WIDTH - 250, 25))

    chord_text = FONT_MED.render(predicted_chord_display, True, BLUE)
    SCREEN.blit(chord_text, (WIDTH - 250, 50))


    # ---------------- DRAW PIANO ----------------

    visible_oct1 = 4 + octave_offset
    visible_oct2 = visible_oct1 + 1

    for i, key in enumerate(WHITE_KEYS):

        octave = visible_oct1 if i < 7 else visible_oct2
        note_name = f"{key}{octave}"

        rect = pygame.Rect(
            i * WHITE_KEY_WIDTH,
            PIANO_TOP,
            WHITE_KEY_WIDTH,
            WHITE_KEY_HEIGHT
        )

        color = BLUE if note_name in notes_pressed else WHITE
        pygame.draw.rect(SCREEN, color, rect)
        pygame.draw.rect(SCREEN, BLACK, rect, 2)

        octave_index = 1 if i < 7 else 2
        base_key = key if octave_index == 1 else key + "2"
        if base_key in KEYBOARD_LABELS:
            label = FONT_SMALL.render(KEYBOARD_LABELS[base_key], True, BLACK)
            SCREEN.blit(label, (rect.centerx - 6, rect.bottom - 20))

    for i, key in enumerate(BLACK_KEYS):
        if key:
            octave = visible_oct1 if i < 7 else visible_oct2
            note_name = f"{key}{octave}"
            rect = pygame.Rect(
                i * WHITE_KEY_WIDTH + 0.7 * WHITE_KEY_WIDTH,
                PIANO_TOP,
                BLACK_KEY_WIDTH,
                BLACK_KEY_HEIGHT
            )

            color = BLUE if note_name in notes_pressed else BLACK
            pygame.draw.rect(SCREEN, color, rect)

            octave_index = 1 if i < 7 else 2
            base_key = key if octave_index == 1 else key + "2"
            if base_key in KEYBOARD_LABELS:
                label = FONT_SMALL.render(KEYBOARD_LABELS[base_key], True, WHITE)
                SCREEN.blit(label, (rect.centerx - 6, rect.bottom - 18))

    pygame.display.flip()


    # ---------------- EVENTS ----------------

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if BTN_MINUS.collidepoint(event.pos):
                tempo_index = max(0, tempo_index - 1)
                tempo = TEMPO_OPTIONS[tempo_index]
                engine.set_tempo(tempo)

            if BTN_PLUS.collidepoint(event.pos):

                tempo_index = min(
                    len(TEMPO_OPTIONS) - 1,
                    tempo_index + 1
                )

                tempo = TEMPO_OPTIONS[tempo_index]
                engine.set_tempo(tempo)

        elif event.type == pygame.KEYDOWN:
            key_name = pygame.key.name(event.key)

            if key_name == "z":
                octave_offset = max(OCTAVE_MIN, octave_offset - 1)

                # stop all active notes
                for note in list(notes_pressed):
                    NOTE_CHANNELS[note].stop()
                    play_top_note()

                notes_pressed.clear()
                notes_played.clear()


            elif key_name == "x":
                octave_offset = min(OCTAVE_MAX, octave_offset + 1)

                # stop all active notes
                for note in list(notes_pressed):
                    NOTE_CHANNELS[note].stop()
                    play_top_note()
                notes_pressed.clear()
                notes_played.clear()

            elif key_name in KEYBOARD_MAP:

                base_note = KEYBOARD_MAP[key_name]

                note = shift_note_octave(
                    base_note,
                    octave_offset
                )

                if note and note not in notes_pressed:
                    notes_pressed[note] = current_time
                    play_top_note()

        elif event.type == pygame.KEYUP:
            key_name = pygame.key.name(event.key)

            if key_name in KEYBOARD_MAP:

                base_note = KEYBOARD_MAP[key_name]

                note = shift_note_octave(
                    base_note,
                    octave_offset
                )

                if note in notes_pressed:

                    start = notes_pressed.pop(note)

                    notes_played.append(
                        (note, start, current_time)
                    )

                    NOTE_CHANNELS[note].stop()
                    play_top_note()


pygame.quit()

if midi_input:
    midi_input.close()

pygame.midi.quit()