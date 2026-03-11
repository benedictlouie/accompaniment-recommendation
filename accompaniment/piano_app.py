import pygame
import pygame.midi
import time

from engines.factory import create_engine
from utils.constants import SAMPLE_RATE, NOTE_FREQS, ROOTS, FONT_BIG, FONT_MED, FONT_SMALL, BLACK, GRAY, BLUE, DARK_GRAY, WHITE, BLACK_KEYS, WHITE_KEYS, KEYBOARD_LABELS, KEYBOARD_MAP
from accompaniment.accompaniment_system import AccompanimentSystem, SoundGenerator
from utils.metronome import Metronome

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
NOTE_CHANNELS = {note: pygame.mixer.Channel(i) for i, note in enumerate(NOTE_FREQS)}
NOTE_SOUNDS = SoundGenerator().note_sounds

def play_top_note():
    for ch in NOTE_CHANNELS.values():
        ch.stop()
    if not notes_pressed:
        return
    highest_note = max(notes_pressed.keys(), key=lambda n: NOTE_FREQS[n])
    NOTE_CHANNELS[highest_note].play(NOTE_SOUNDS[highest_note], loops=-1)

# --------------------------------------------------
# MIDI INPUT
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
# TEMPO / METRONOME
# --------------------------------------------------
tempo = 100
metronome = Metronome(tempo, WIDTH)

# --------------------------------------------------
# ENGINE
# --------------------------------------------------
ENGINE_TYPE = "transformer"
engine = create_engine(ENGINE_TYPE, tempo)
accompaniment_system = AccompanimentSystem()

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

    # ---------------- MIDI ----------------
    if midi_input and midi_input.poll():
        midi_events = midi_input.read(10)
        for event in midi_events:
            data = event[0]
            status, note_num, velocity = data[0], data[1], data[2]
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

    # ---------------- METRONOME CLOCK ----------------
    beat_trigger, beat_start_time = metronome.update()
    if beat_trigger:
        current_beat = metronome.current_beat
        chord, duration = engine.process_beat(notes_played, beat_start_time, current_beat)
        # mute if chord == "N"
        metronome.mute(chord != "N")
        metronome.click()
        melody = engine.last_bar
        accompaniment_system.play_beat(melody, chord, tempo, current_beat)
        if chord:
            predicted_chord_display = chord
        notes_played.clear()
        metronome.advance()

    # ---------------- DRAW ----------------
    SCREEN.fill(BLACK)
    pygame.draw.rect(SCREEN, DARK_GRAY, (0, 0, WIDTH, TOP_BAR_HEIGHT))

    # draw current range (z/x to change)
    range_text = FONT_SMALL.render(f"Range {get_visible_range()} (z/x to change)", True, WHITE)
    SCREEN.blit(range_text, (WIDTH//2 - 110, 70))

    # draw metronome (tempo + buttons + beat dots)
    metronome.draw(SCREEN)

    # Chord display
    chord_label = FONT_SMALL.render("Predicted Chord", True, GRAY)
    SCREEN.blit(chord_label, (WIDTH - 250, 25))
    chord_text = FONT_MED.render(predicted_chord_display, True, BLUE)
    SCREEN.blit(chord_text, (WIDTH - 250, 50))

    # Piano drawing
    visible_oct1 = 4 + octave_offset
    visible_oct2 = visible_oct1 + 1
    for i, key in enumerate(WHITE_KEYS):
        octave = visible_oct1 if i < 7 else visible_oct2
        note_name = f"{key}{octave}"
        rect = pygame.Rect(i * (WIDTH // len(WHITE_KEYS)), PIANO_TOP, WIDTH // len(WHITE_KEYS), PIANO_HEIGHT)
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
            rect = pygame.Rect(i * (WIDTH // len(WHITE_KEYS)) + 0.7 * (WIDTH // len(WHITE_KEYS)), PIANO_TOP,
                               (WIDTH // len(WHITE_KEYS)) // 2, int(PIANO_HEIGHT * 0.6))
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
        if metronome.handle_event(event):
            tempo = metronome.tempo
            engine.set_tempo(tempo)
        elif event.type == pygame.KEYDOWN:
            key_name = pygame.key.name(event.key)
            if key_name == "z":
                octave_offset = max(OCTAVE_MIN, octave_offset - 1)
                for note in list(notes_pressed):
                    NOTE_CHANNELS[note].stop()
                    play_top_note()
                notes_pressed.clear()
                notes_played.clear()
            elif key_name == "x":
                octave_offset = min(OCTAVE_MAX, octave_offset + 1)
                for note in list(notes_pressed):
                    NOTE_CHANNELS[note].stop()
                    play_top_note()
                notes_pressed.clear()
                notes_played.clear()
            elif key_name in KEYBOARD_MAP:
                base_note = KEYBOARD_MAP[key_name]
                note = shift_note_octave(base_note, octave_offset)
                if note and note not in notes_pressed:
                    notes_pressed[note] = current_time
                    play_top_note()
        elif event.type == pygame.KEYUP:
            key_name = pygame.key.name(event.key)
            if key_name in KEYBOARD_MAP:
                base_note = KEYBOARD_MAP[key_name]
                note = shift_note_octave(base_note, octave_offset)
                if note in notes_pressed:
                    start = notes_pressed.pop(note)
                    notes_played.append((note, start, current_time))
                    NOTE_CHANNELS[note].stop()
                    play_top_note()

pygame.quit()
if midi_input:
    midi_input.close()
pygame.midi.quit()