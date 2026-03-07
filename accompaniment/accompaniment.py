import numpy as np
import random
import pygame
import time
import fluidsynth
from utils.constants import CHORD_TO_TETRAD, NOTE_FREQS, STEPS_PER_BAR, STEPS_PER_BEAT, SAMPLE_RATE
import threading

# --------------------------------
# FLUIDSYNTH ENGINE (GLOBAL)
# --------------------------------

def get_fs(drum_sf=None, guitar_sf=None, piano_sf=None, bass_sf=None):
    fs = fluidsynth.Synth(gain=3)
    fs.start(driver="coreaudio")
    if drum_sf is not None:
        sfid = fs.sfload(drum_sf)
        fs.program_select(9, sfid, 0, 0)
    if guitar_sf is not None:
        sfid = fs.sfload(guitar_sf)
        fs.program_select(0, sfid, 0, 0)
    if piano_sf is not None:
        sfid = fs.sfload(piano_sf)
        fs.program_select(1, sfid, 0, 0)
    if bass_sf is not None:
        sfid = fs.sfload(piano_sf)
        fs.program_select(2, sfid, 0, 0)
    return fs

# Dummy silent sound (to keep callers working)
pygame.mixer.pre_init(SAMPLE_RATE, -16, 2, 256)
pygame.init()

def apply_lowpass(wave, cutoff):
    alpha = cutoff / (cutoff + SAMPLE_RATE)
    filtered = np.zeros_like(wave)
    for i in range(1, len(wave)):
        filtered[i] = filtered[i-1] + alpha * (wave[i] - filtered[i-1])
    return filtered

def generate_additive_wave(freq, t, harmonic_amplitudes):
    """
    Sum of sine partials at integer multiples of fundamental.
    harmonic_amplitudes: list of amplitudes starting from 1st harmonic.
    """
    wave = np.zeros_like(t)
    for n, amp in enumerate(harmonic_amplitudes, start=1):
        wave += amp * np.sin(2 * np.pi * freq * n * t)
    # Normalize to prevent clipping
    wave /= np.max(np.abs(wave)) if np.max(np.abs(wave)) > 0 else 1
    return wave

def generate_note_sound(freq, duration=1.0, pan=0.0, velocity=1.0, timbre="piano"):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)

    # -------------------------
    # TIMBRE DEFINITIONS
    # -------------------------

    if timbre == "piano":
        # More weight on low and mid harmonics
        harmonics = [1.0, 0.8, 0.6, 0.4, 0.3, 0.2]
        wave = generate_additive_wave(freq, t, harmonics)
        # Add a bit of noise for hammer sound (piano transient)
        noise = 0.01 * np.random.normal(size=len(t))
        wave += noise
        cutoff = 3500

    elif timbre == "guitar":
        # Guitar has many partials with slower decay
        harmonics = [1.0, 0.7, 0.5, 0.3, 0.2, 0.15, 0.1]
        wave = generate_additive_wave(freq, t, harmonics)
        # Slight pluck envelope shaping (quick decay)
        envelope = np.exp(-5 * t)
        wave *= envelope
        cutoff = 2500

    elif timbre == "pad":
        # Pad sounds use lots of harmonics and detuned layers
        base_harmonics = [1.0, 0.9, 0.8, 0.6, 0.4]
        wave = generate_additive_wave(freq, t, base_harmonics)
        # Add slow detune for width
        detune = generate_additive_wave(freq * 1.005, t, base_harmonics) * 0.5
        wave += detune
        cutoff = 1200

    elif timbre == "retro":
        # Retro (square-like) built from odd harmonics
        harmonics = [1.0/(2*k-1) for k in range(1, 10)]
        wave = generate_additive_wave(freq, t, harmonics)
        cutoff = 4000

    elif timbre == "orchestral":
        # Orchestral rich partial series (more components)
        harmonics = [1.0, 0.7, 0.55, 0.4, 0.3, 0.2, 0.15]
        wave = generate_additive_wave(freq, t, harmonics)
        cutoff = 3000

    elif timbre == "lofi":
        # Lofi uses simpler harmonics + noise
        harmonics = [1.0, 0.5, 0.3]
        wave = generate_additive_wave(freq, t, harmonics)
        noise = np.random.normal(0, 0.03, len(t))
        wave += noise
        cutoff = 1500

    else:
        # Default: simple fundamental sine
        wave = np.sin(2 * np.pi * freq * t)
        cutoff = 3000

    # -------------------------
    # FILTER
    # -------------------------
    wave = apply_lowpass(wave, cutoff)

    # -------------------------
    # ENVELOPE
    # -------------------------
    attack = min(int(0.01 * SAMPLE_RATE), len(wave) // 2)
    release = min(int(0.2 * SAMPLE_RATE), len(wave) // 2)
    envelope = np.ones_like(wave)
    envelope[:attack] *= np.linspace(0, 1, attack)
    envelope[-release:] *= np.linspace(1, 0, release)

    wave *= envelope * velocity * 0.4

    # -------------------------
    # STEREO PAN
    # -------------------------
    left = wave * (1 - pan)
    right = wave * (1 + pan)

    stereo = np.column_stack([left, right])
    stereo = np.int16(stereo * 32767)

    return pygame.sndarray.make_sound(stereo)

NOTE_SOUNDS = { note: generate_note_sound(freq * 2) for note, freq in NOTE_FREQS.items() }

def play_harmony(chord_name, duration, fs, creative=True):
    
    if chord_name not in CHORD_TO_TETRAD:
        return

    notes = CHORD_TO_TETRAD[chord_name]

    if creative:
        inversion = random.choice([0, 1, 2])
        notes = notes[inversion:] + notes[:inversion]
        notes = [n + random.choice([0, 12]) for n in notes]

    for i, midi in enumerate(notes):
        if midi < 10:
            continue

        velocity = int(random.uniform(0.9, 1.0) * 127)

        delay = 0
        if creative:
            delay = min(1, i) * duration * random.choice([0, 0.5])

        def play_note(m=midi, vel=velocity):
            fs.noteon(0, m, vel)

            def stop_note():
                time.sleep(duration)
                fs.noteoff(0, m)

            threading.Thread(target=stop_note, daemon=True).start()

        if delay > 0:
            threading.Timer(delay, play_note).start()
        else:
            play_note()

def snap_to_chord(target, chord):
    best = None
    best_dist = 1e9

    for octave in (-12, 0, 12):
        for n in chord:
            candidate = n + octave
            d = abs(candidate - target)
            if d < best_dist:
                best_dist = d
                best = candidate

    return best

ACTIVE_NOTES = {
    "guitar": set(),
    "piano": set(),
    "bass": set(),
}

def play_harmony_nn(chord_name, loop, beat_index, fs, instrument, bpm):

    if chord_name not in CHORD_TO_TETRAD:
        return

    chord = CHORD_TO_TETRAD[chord_name]

    channel = {"guitar":0, "piano":1, "bass":2}[instrument]
    velocity = {"guitar":85, "piano": 100, "bass":70}[instrument]

    if instrument == "piano":
        chord = [n+12 for n in chord]
    if instrument == "bass":
        chord = [n-12 for n in chord]

    seconds_per_beat = 60.0 / bpm
    step_duration = seconds_per_beat / STEPS_PER_BEAT

    start_step = beat_index * STEPS_PER_BEAT

    for sub in range(STEPS_PER_BEAT):

        step = start_step + sub

        if loop is None or step >= len(loop):
            break

        delay = sub * step_duration

        notes_now = set()

        for interval in loop[step]:

            if interval < 0:
                continue

            target = chord[0] + interval
            midi = snap_to_chord(target, chord)

            notes_now.add(midi)

        def process_step(notes_now=notes_now):

            active = ACTIVE_NOTES[instrument]

            # notes to stop
            for n in active - notes_now:
                fs.noteoff(channel, n)

            # notes to start
            for n in notes_now - active:
                fs.noteon(channel, n, velocity)

            ACTIVE_NOTES[instrument] = notes_now

        threading.Timer(delay, process_step).start()

def play_drum_loop(drum_loop, fs, bpm):
    """
    Play a (16,128) drum loop non-blocking, max velocity, adjustable gain.
    """

    def _play_loop(loop):

        step_duration = 60 / bpm / STEPS_PER_BEAT  # 16th notes

        for step in loop[:STEPS_PER_BAR]:
            notes = np.where(step > 0)[0]

            for n in notes:
                fs.noteon(9, int(n), 127)

            time.sleep(step_duration)

            for n in notes:
                fs.noteoff(9, int(n))

    # run playback in a separate thread (non-blocking)
    t = threading.Thread(target=_play_loop, args=(drum_loop,), daemon=True)
    t.start()

    return drum_loop