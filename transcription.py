import time
import numpy as np
import sounddevice as sd
import pygame
import librosa
import collections
from liveaudio.LivePyin import LivePyin
from liveaudio.buffers import CircularBuffer

# =============================
# METRONOME SETTINGS
# =============================
BPM = 100   # Slower default
SUBDIV = 4

def update_timing():
    global SECONDS_PER_BEAT, SUBDIV_DURATION
    SECONDS_PER_BEAT = 60 / BPM
    SUBDIV_DURATION = SECONDS_PER_BEAT / SUBDIV

update_timing()

# =============================
# AUDIO SETTINGS
# =============================
SAMPLE_RATE = 44100
FRAME_LENGTH = 2048
HOP_LENGTH = 512
VOLUME_THRESHOLD = 0.01

fmin = librosa.note_to_hz('C2')
fmax = librosa.note_to_hz('C6')

current_pitch = None
current_amplitude = 0.0

# =============================
# PITCH TRACKER
# =============================
lpyin = LivePyin(
    fmin, fmax,
    sr=SAMPLE_RATE,
    frame_length=FRAME_LENGTH,
    hop_length=HOP_LENGTH,
)

buffer = CircularBuffer(FRAME_LENGTH, HOP_LENGTH)

# =============================
# VISUAL BUFFERS
# =============================
waveform_buffer = collections.deque(maxlen=2048)

MIDI_MIN = 36
MIDI_MAX = 84
SPEC_HEIGHT = MIDI_MAX - MIDI_MIN + 1
SPEC_WIDTH = 128  # number of 16th notes visible
spectrogram = np.zeros((SPEC_HEIGHT, SPEC_WIDTH))

# =============================
# MIDI HELPERS
# =============================
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F',
              'F#', 'G', 'G#', 'A', 'A#', 'B']

def midi_to_note_name(midi):
    octave = (midi // 12) - 1
    return f"{NOTE_NAMES[midi % 12]}{octave}"

# =============================
# AUDIO CALLBACK
# =============================
def audio_callback(indata, frames, time_info, status):
    global current_pitch, current_amplitude

    samples = indata[:, 0]
    waveform_buffer.extend(samples.tolist())
    current_amplitude = np.sqrt(np.mean(samples**2))

    buffer.push(samples)

    if buffer.full:
        block = buffer.get()
        f0, voiced_flag, _ = lpyin.step(block)
        if voiced_flag and np.isfinite(f0):
            current_pitch = f0
        else:
            current_pitch = None

# =============================
# PYGAME INIT
# =============================
pygame.init()
WIDTH, HEIGHT = 1100, 650
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("16th Note Quantised Pitch Visualiser")

clock = pygame.time.Clock()
pygame.mixer.init()
click = pygame.mixer.Sound("utils/click.wav")

font_big = pygame.font.SysFont("Arial", 40)
font_small = pygame.font.SysFont("Arial", 22)

# =============================
# DRAW FUNCTIONS
# =============================
def draw_waveform():
    if len(waveform_buffer) < 2:
        return

    data = np.array(waveform_buffer)
    data = data / (np.max(np.abs(data)) + 1e-6)

    mid = HEIGHT // 4
    scale_x = WIDTH / len(data)
    scale_y = 80

    points = [(i * scale_x, mid - sample * scale_y)
              for i, sample in enumerate(data)]

    pygame.draw.lines(screen, (0, 255, 0), False, points, 1)


def draw_spectrogram():
    col_width = WIDTH / SPEC_WIDTH
    row_height = (HEIGHT//2) / SPEC_HEIGHT

    for x in range(SPEC_WIDTH):
        for y in range(SPEC_HEIGHT):
            if spectrogram[y, x] > 0:
                rect = pygame.Rect(
                    x * col_width,
                    HEIGHT//2 + y * row_height,
                    col_width,
                    row_height
                )
                pygame.draw.rect(screen, (255, 140, 0), rect)


def advance_16th():
    global spectrogram

    spectrogram = np.roll(spectrogram, -1, axis=1)
    spectrogram[:, -1] = 0

    if current_amplitude < VOLUME_THRESHOLD:
        return

    if current_pitch is None:
        return

    midi = int(round(69 + 12*np.log2(current_pitch/440.0)))

    if MIDI_MIN <= midi <= MIDI_MAX:
        row = MIDI_MAX - midi
        spectrogram[row, -1] = 1


# =============================
# AUDIO STREAM
# =============================
stream = sd.InputStream(
    channels=1,
    samplerate=SAMPLE_RATE,
    blocksize=HOP_LENGTH,
    callback=audio_callback,
)

# =============================
# MAIN LOOP
# =============================
start_time = time.time()
subdiv_index = 0
beat_count = 0
beat_results = []

running = True

with stream:
    while running:
        screen.fill((0, 0, 0))
        now = time.time()

        # --- EVENT HANDLING ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    BPM += 5
                    update_timing()
                if event.key == pygame.K_DOWN:
                    BPM = max(20, BPM - 5)
                    update_timing()
                if event.key == pygame.K_ESCAPE:
                    running = False

        # --- 16TH NOTE CLOCK ---
        next_tick = start_time + subdiv_index * SUBDIV_DURATION

        if now >= next_tick:

            # Play click on downbeat
            if subdiv_index % SUBDIV == 0:
                click.play()

            # --------------------------
            # Capture pitch for this 16th
            # --------------------------
            if current_amplitude < VOLUME_THRESHOLD:
                beat_results.append("quiet")
            else:
                if current_pitch is None:
                    beat_results.append("no pitch")
                else:
                    midi = int(round(69 + 12*np.log2(current_pitch/440.0)))
                    beat_results.append(midi_to_note_name(midi))

            # Advance spectrogram
            advance_16th()

            subdiv_index += 1

            # --------------------------
            # If full beat completed
            # --------------------------
            if len(beat_results) == SUBDIV:
                print(f"Beat {beat_count + 1}: {beat_results}")
                beat_results = []
                beat_count += 1

        # --- DRAW ---
        draw_waveform()
        draw_spectrogram()

        # Current note display
        if current_pitch and current_amplitude > VOLUME_THRESHOLD:
            midi = int(round(69 + 12*np.log2(current_pitch/440.0)))
            note_name = midi_to_note_name(midi)
            text = font_big.render(note_name, True, (255,255,255))
            screen.blit(text, (20, 20))

        bpm_text = font_small.render(f"BPM: {BPM}", True, (200,200,200))
        screen.blit(bpm_text, (20, 80))

        pygame.display.flip()
        clock.tick(60)

pygame.quit()
