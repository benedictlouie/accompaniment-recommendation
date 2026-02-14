import time
import numpy as np
import sounddevice as sd
from liveaudio.LivePyin import LivePyin
from liveaudio.buffers import CircularBuffer

# 🕰 Metronome settings
BPM = 120
SECONDS_PER_BEAT = 60 / BPM
SUBDIV = 4  # number of subdivisions per beat
SUBDIV_DURATION = SECONDS_PER_BEAT / SUBDIV

# Audio settings
SAMPLE_RATE = 44100
FRAME_LENGTH = 4096
HOP_LENGTH = 1024

# Volume threshold (RMS)
VOLUME_THRESHOLD = 0.01

# Pitch range for typical vocal range (C2–C6)
import librosa
fmin = librosa.note_to_hz('C2')
fmax = librosa.note_to_hz('C6')

current_pitch = None
current_amplitude = 0.0

# Setup pitch tracker
lpyin = LivePyin(
    fmin, fmax,
    sr=SAMPLE_RATE,
    frame_length=FRAME_LENGTH,
    hop_length=HOP_LENGTH,
    n_bins_per_semitone=10,
    max_semitones_per_frame=8,
)

buffer = CircularBuffer(FRAME_LENGTH, HOP_LENGTH)

def audio_callback(indata, frames, time_info, status):
    global current_pitch, current_amplitude

    samples = indata[:, 0]
    buffer.push(samples)

    # compute RMS amplitude for loudness check
    current_amplitude = np.sqrt(np.mean(samples**2))

    if buffer.full:
        block = buffer.get()
        f0, voiced_flag, voiced_prob = lpyin.step(block)
        if voiced_flag and np.isfinite(f0):
            current_pitch = f0
        else:
            current_pitch = None

# convert MIDI number to note name
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
def midi_to_note_name(midi):
    octave = (midi // 12) - 1
    return f"{NOTE_NAMES[midi % 12]}{octave}"

# start audio stream
stream = sd.InputStream(
    channels=1,
    samplerate=SAMPLE_RATE,
    blocksize=HOP_LENGTH,
    callback=audio_callback,
)

print(f"Listening… BPM={BPM}, threshold={VOLUME_THRESHOLD:.3f} (Ctrl‑C to stop)")

beat_count = 0
start_time = time.time()

try:
    with stream:
        while True:
            # wait until start of this beat
            beat_start = start_time + beat_count * SECONDS_PER_BEAT
            wait = beat_start - time.time()
            if wait > 0:
                time.sleep(wait)

            results = []
            for i in range(SUBDIV):
                # for each subdivision of the beat
                subdiv_time = beat_start + i * SUBDIV_DURATION
                wait = subdiv_time - time.time()
                if wait > 0:
                    time.sleep(wait)

                # capture the current state
                if current_amplitude < VOLUME_THRESHOLD:
                    results.append("quiet")
                else:
                    if current_pitch is None:
                        results.append("no pitch")
                    else:
                        midi = int(round(69 + 12*np.log2(current_pitch/440.0)))
                        results.append(midi_to_note_name(midi))

            # print all 4 subdivisions in one line
            print(f"Beat {beat_count+1}: {results}")

            beat_count += 1

except KeyboardInterrupt:
    print("\nStopped.")
