import numpy as np
import random
import pygame
import time
import fluidsynth
import threading
from utils.constants import CHORD_TO_TETRAD, NOTE_FREQS, STEPS_PER_BAR, STEPS_PER_BEAT, SAMPLE_RATE, BEATS_PER_BAR
from accompaniment.nn import get_all_loops

# --------------------------------------------------
# AUDIO ENGINE
# --------------------------------------------------

class AudioEngine:

    def __init__(self, drum_sf=None, guitar_sf=None, piano_sf=None, bass_sf=None):

        self.fs = fluidsynth.Synth(gain=3)
        self.fs.start(driver="coreaudio")

        if drum_sf:
            sfid = self.fs.sfload(drum_sf)
            self.fs.program_select(9, sfid, 0, 0)

        if guitar_sf:
            sfid = self.fs.sfload(guitar_sf)
            self.fs.program_select(0, sfid, 0, 0)

        if piano_sf:
            sfid = self.fs.sfload(piano_sf)
            self.fs.program_select(1, sfid, 0, 0)

        if bass_sf:
            sfid = self.fs.sfload(bass_sf)
            self.fs.program_select(2, sfid, 0, 0)

    def note_on(self, channel, midi, velocity):
        self.fs.noteon(channel, midi, velocity)

    def note_off(self, channel, midi):
        self.fs.noteoff(channel, midi)

# --------------------------------------------------
# SOUND GENERATOR
# --------------------------------------------------

class SoundGenerator:

    def __init__(self):
        pygame.mixer.pre_init(SAMPLE_RATE, -16, 2, 256)
        pygame.init()

        self.note_sounds = {
            note: self.generate_note_sound(freq * 2)
            for note, freq in NOTE_FREQS.items()
        }

    def apply_lowpass(self, wave, cutoff):

        alpha = cutoff / (cutoff + SAMPLE_RATE)
        filtered = np.zeros_like(wave)

        for i in range(1, len(wave)):
            filtered[i] = filtered[i - 1] + alpha * (wave[i] - filtered[i - 1])

        return filtered

    def generate_additive_wave(self, freq, t, harmonic_amplitudes):

        wave = np.zeros_like(t)

        for n, amp in enumerate(harmonic_amplitudes, start=1):
            wave += amp * np.sin(2 * np.pi * freq * n * t)

        wave /= np.max(np.abs(wave)) if np.max(np.abs(wave)) > 0 else 1
        return wave

    def generate_note_sound(self, freq, duration=1.0, pan=0.0, velocity=0.7, timbre="piano"):

        t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)

        if timbre == "piano":
            harmonics = [1.0, 0.8, 0.6, 0.4, 0.3, 0.2]
            wave = self.generate_additive_wave(freq, t, harmonics)
            wave += 0.01 * np.random.normal(size=len(t))
            cutoff = 3500

        elif timbre == "guitar":
            harmonics = [1.0, 0.7, 0.5, 0.3, 0.2]
            wave = self.generate_additive_wave(freq, t, harmonics)
            wave *= np.exp(-5 * t)
            cutoff = 2500

        else:
            wave = np.sin(2 * np.pi * freq * t)
            cutoff = 3000
        
        wave = self.apply_lowpass(wave, cutoff)

        attack = min(int(0.01 * SAMPLE_RATE), len(wave) // 2)
        release = min(int(0.2 * SAMPLE_RATE), len(wave) // 2)

        envelope = np.ones_like(wave)
        envelope[:attack] *= np.linspace(0, 1, attack)
        envelope[-release:] *= np.linspace(1, 0, release)

        wave *= envelope * velocity * 0.4

        left = wave * (1 - pan)
        right = wave * (1 + pan)

        stereo = np.column_stack([left, right])
        stereo = np.int16(stereo * 32767)

        return pygame.sndarray.make_sound(stereo)



# --------------------------------------------------
# HARMONY PLAYER
# --------------------------------------------------

class HarmonyPlayer:

    def __init__(self, engine):

        self.engine = engine

        self.active_notes = {
            "guitar": set(),
            "piano": set(),
            "bass": set(),
        }

    # --------------------------------------------------
    # SIMPLE GUITAR HARMONY
    # --------------------------------------------------

    def play_simple_guitar(self, chord_name, duration):

        if chord_name not in CHORD_TO_TETRAD:
            return

        notes = CHORD_TO_TETRAD[chord_name]

        inversion = random.choice([0, 1, 2])
        notes = notes[inversion:] + notes[:inversion]

        for i, midi in enumerate(notes):

            velocity = int(random.uniform(0.9, 1.0) * 127)

            delay = min(1, i) * duration * random.choice([0, 0.5])

            def play_note(m=midi, vel=velocity):

                self.engine.note_on(0, m, vel)

                def stop():
                    time.sleep(duration)
                    self.engine.note_off(0, m)

                threading.Thread(target=stop, daemon=True).start()

            threading.Timer(delay, play_note).start()

    # --------------------------------------------------
    # SNAP NOTE TO CHORD
    # --------------------------------------------------

    def snap_to_chord(self, target, chord):

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

    # --------------------------------------------------
    # NN HARMONY
    # --------------------------------------------------

    def play_harmony_nn(self, chord_name, loop, beat_index, instrument, bpm):

        if chord_name not in CHORD_TO_TETRAD:
            return
        
        if loop is None:
            return

        chord = CHORD_TO_TETRAD[chord_name]

        channel = {"guitar":0, "piano":1, "bass":2}[instrument]
        velocity = {"guitar":85, "piano":100, "bass":127}[instrument]

        if instrument == "piano":
            chord = [n+12 for n in chord]

        seconds_per_beat = 60.0 / bpm
        step_duration = seconds_per_beat / STEPS_PER_BEAT

        start_step = beat_index * STEPS_PER_BEAT

        for sub in range(STEPS_PER_BEAT):

            step = start_step + sub

            if step >= len(loop):
                break

            delay = sub * step_duration

            notes_now = set()

            for interval in loop[step]:

                if interval < 0:
                    continue

                target = chord[0] + interval
                midi = self.snap_to_chord(target, chord)

                notes_now.add(midi)

            def process_step(notes_now=notes_now):

                active = self.active_notes[instrument]

                for n in active - notes_now:
                    self.engine.note_off(channel, n)

                for n in notes_now - active:
                    self.engine.note_on(channel, n, velocity)

                self.active_notes[instrument] = notes_now

            threading.Timer(delay, process_step).start()


# --------------------------------------------------
# DRUM PLAYER
# --------------------------------------------------

class DrumPlayer:

    def __init__(self, engine):
        self.engine = engine

    def play_loop(self, drum_loop, bpm):

        def _play():

            step_duration = 60 / bpm / STEPS_PER_BEAT

            for step in drum_loop[:STEPS_PER_BAR]:

                notes = np.where(step > 0)[0]

                for n in notes:
                    self.engine.note_on(9, int(n), 127)

                time.sleep(step_duration)

                for n in notes:
                    self.engine.note_off(9, int(n))

        threading.Thread(target=_play, daemon=True).start()


# --------------------------------------------------
# MUSIC SYSTEM
# --------------------------------------------------

class AccompanimentSystem:

    def __init__(self, symphony=True):

        pygame.mixer.pre_init(SAMPLE_RATE, -16, 2, 256)
        pygame.init()

        self.symphony = symphony

        self.engine = AudioEngine(
            drum_sf="soundfonts/The_Definitive_Perfect_Drums_Soundfount_V1___1-12_.sf2",
            guitar_sf="soundfonts/Guitar.sf2",
            piano_sf="soundfonts/UprightPianoKW-small-bright-20190703.sf2",
            bass_sf="soundfonts/FingerBassYR 20190930.sf2"
        )

        self.harmony = HarmonyPlayer(self.engine)
        self.drums = DrumPlayer(self.engine)

        self.prev_drum_loop = None

        # cache loops for the current bar
        self.guitar_loop = None
        self.piano_loop = None
        self.bass_loop = None

    # --------------------------------------------------

    def play_beat(self, melody, chord, tempo, current_beat):

        beat_in_bar = current_beat % BEATS_PER_BAR

        # --------------------------------------------------
        # SYMPHONY MODE
        # --------------------------------------------------

        if self.symphony:

            # Generate loops at start of bar
            if beat_in_bar == 0:

                drum_loop, piano_loop, guitar_loop, bass_loop = get_all_loops(melody)

                if np.any(guitar_loop != -1):
                    self.guitar_loop = guitar_loop
                if np.any(piano_loop != -1):
                    self.piano_loop = piano_loop
                if np.any(bass_loop != -1):
                    self.bass_loop = bass_loop

                drum_steps, _ = np.nonzero(drum_loop)
                if len(drum_steps) < 10:
                    drum_loop = self.prev_drum_loop

                if chord != "N" and drum_loop is not None:
                    self.drums.play_loop(drum_loop, tempo)
                    self.prev_drum_loop = drum_loop

            # HARMONY EVERY BEAT
            if chord and chord != "N":
                self.harmony.play_harmony_nn(chord, self.guitar_loop, beat_in_bar, "guitar", tempo)
                self.harmony.play_harmony_nn(chord, self.piano_loop, beat_in_bar, "piano", tempo)
                self.harmony.play_harmony_nn(chord, self.bass_loop, beat_in_bar, "bass", tempo)

        # --------------------------------------------------
        # SIMPLE MODE
        # --------------------------------------------------

        else:
            if chord:
                duration = 60 / tempo
                self.harmony.play_simple_guitar(chord, duration)