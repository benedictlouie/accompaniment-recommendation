import numpy as np
import pygame
from utils.constants import CHORD_TO_TETRAD, NOTE_FREQS

def generate_note_sound(freq, duration=1.0):
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = 0.3 * np.sin(2 * np.pi * freq * t)
    wave = np.int16(wave * 32767)
    stereo = np.column_stack([wave, wave])
    return pygame.sndarray.make_sound(stereo)

NOTE_SOUNDS = {
    note: generate_note_sound(freq * 2)
    for note, freq in NOTE_FREQS.items()
}

def play_harmony(chord_name, duration, harmony_channels):
    if chord_name not in CHORD_TO_TETRAD:
        return

    for i, midi in enumerate(CHORD_TO_TETRAD[chord_name]):
        if midi < 10: continue
        freq = 440 * 2 ** (1 + (midi - 69) / 12)
        sound = generate_note_sound(freq, duration)
        harmony_channels[i].play(sound)