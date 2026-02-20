import numpy as np
import random
import pygame
import time
from utils.constants import CHORD_TO_TETRAD, NOTE_FREQS

SAMPLE_RATE = 44100

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

def play_harmony(chord_name, duration, harmony_channels, creative=True, timbre="piano"):
    if chord_name not in CHORD_TO_TETRAD:
        return

    notes = CHORD_TO_TETRAD[chord_name]

    if creative:
        inversion = random.choice([0, 1, 2])
        notes = notes[inversion:] + notes[:inversion]
        notes = [n + random.choice([-12, 0]) for n in notes]

    for i, midi in enumerate(notes):
        if midi < 10:
            continue

        base_freq = 440 * 2 ** ((midi - 69) / 12)

        velocity = random.uniform(0.6, 1.0)
        pan = random.uniform(-0.7, 0.7)

        # 🎲 random delay in first half of duration
        delay = min(1, i) * duration * random.choice([0, 0.4])
        delay_samples = int(delay * SAMPLE_RATE)

        sound = generate_note_sound(
            base_freq,
            duration,
            pan=pan,
            velocity=velocity,
            timbre=timbre
        )

        sound_array = pygame.sndarray.array(sound)

        if sound_array.ndim == 1:
            silence = np.zeros(delay_samples, dtype=sound_array.dtype)
            new_array = np.concatenate((silence, sound_array))
        else:
            silence = np.zeros((delay_samples, sound_array.shape[1]), dtype=sound_array.dtype)
            new_array = np.vstack((silence, sound_array))

        # 🔹 Convert back to Sound
        delayed_sound = pygame.sndarray.make_sound(new_array)
        harmony_channels[i % len(harmony_channels)].play(delayed_sound)