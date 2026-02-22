import numpy as np
import sounddevice as sd
import librosa
import collections
from liveaudio.LivePyin import LivePyin
from liveaudio.buffers import CircularBuffer
from utils.constants import ROOTS


class Transcriber:
    def __init__(
        self,
        sample_rate=44100,
        frame_length=2048,
        hop_length=512,
        volume_threshold=0.01,
        midi_min=36,
        midi_max=96,
        spec_width=128
    ):
        # =============================
        # AUDIO SETTINGS
        # =============================
        self.SAMPLE_RATE = sample_rate
        self.FRAME_LENGTH = frame_length
        self.HOP_LENGTH = hop_length
        self.VOLUME_THRESHOLD = volume_threshold

        self.current_pitch = None
        self.current_amplitude = 0.0

        # Pitch limits
        self.fmin = librosa.note_to_hz('C2')
        self.fmax = librosa.note_to_hz('C6')

        # Live pyin (original system)
        self.lpyin = LivePyin(
            self.fmin,
            self.fmax,
            sr=self.SAMPLE_RATE,
            frame_length=self.FRAME_LENGTH,
            hop_length=self.HOP_LENGTH,
        )

        self.buffer = CircularBuffer(self.FRAME_LENGTH, self.HOP_LENGTH)

        # =============================
        # OFFLINE BEAT BUFFER
        # =============================
        self.beat_audio_buffer = []

        # =============================
        # VISUAL BUFFERS
        # =============================
        self.waveform_buffer = collections.deque(maxlen=2048)

        self.MIDI_MIN = midi_min
        self.MIDI_MAX = midi_max
        self.SPEC_HEIGHT = self.MIDI_MAX - self.MIDI_MIN + 1
        self.SPEC_WIDTH = spec_width
        self.spectrogram = np.zeros((self.SPEC_HEIGHT, self.SPEC_WIDTH))

        # Stream
        self.stream = sd.InputStream(
            channels=1,
            samplerate=self.SAMPLE_RATE,
            blocksize=self.HOP_LENGTH,
            callback=self.audio_callback,
        )

    # =============================
    # MIDI HELPERS
    # =============================

    def midi_to_note_name(self, midi):
        octave = (midi // 12) - 1
        return f"{str(ROOTS[midi % 12])}{octave}"

    def hz_to_midi(self, hz):
        return int(round(69 + 12 * np.log2(hz / 440.0)))

    # =============================
    # AUDIO CALLBACK
    # =============================
    def audio_callback(self, indata, frames, time_info, status):
        samples = indata[:, 0]

        self.waveform_buffer.extend(samples.tolist())
        self.current_amplitude = np.sqrt(np.mean(samples ** 2))

        # Push to live pitch detector (original behavior)
        self.buffer.push(samples)

        if self.buffer.full:
            block = self.buffer.get()
            f0, voiced_flag, _ = self.lpyin.step(block)

            if voiced_flag and np.isfinite(f0):
                self.current_pitch = f0
            else:
                self.current_pitch = None

        # ALSO accumulate raw audio for offline beat processing
        self.beat_audio_buffer.extend(samples.tolist())

    # =============================
    # ORIGINAL LIVE 16TH (UNCHANGED)
    # =============================
    def capture_16th(self):
        """
        Called by main clock every 16th note.
        Updates spectrogram + returns captured value.
        """

        self.spectrogram = np.roll(self.spectrogram, -1, axis=1)
        self.spectrogram[:, -1] = 0

        if self.current_amplitude < self.VOLUME_THRESHOLD:
            return "quiet"

        if self.current_pitch is None:
            return "no pitch"

        midi = self.hz_to_midi(self.current_pitch)

        if self.MIDI_MIN <= midi <= self.MIDI_MAX:
            row = self.MIDI_MAX - midi
            self.spectrogram[row, -1] = 1

        return self.midi_to_note_name(midi)

    # =============================
    # NEW OFFLINE WHOLE-BEAT METHOD
    # =============================
    def capture_beat_4_16ths(self):
        """
        Called once per beat.
        Processes entire beat audio using librosa.pyin
        and returns 4 MIDI values (one per 16th).
        """

        if len(self.beat_audio_buffer) == 0:
            return ["quiet"] * 4

        audio = np.array(self.beat_audio_buffer)

        # Clear immediately for next beat
        self.beat_audio_buffer = []

        if np.sqrt(np.mean(audio ** 2)) < self.VOLUME_THRESHOLD:
            return ["quiet"] * 4

        # Run full pyin on whole beat
        f0, voiced_flag, _ = librosa.pyin(
            audio,
            fmin=self.fmin,
            fmax=self.fmax,
            sr=self.SAMPLE_RATE,
            frame_length=self.FRAME_LENGTH,
            hop_length=self.HOP_LENGTH,
        )

        if f0 is None:
            return ["no pitch"] * 4

        midi_track = []
        for freq, voiced in zip(f0, voiced_flag):
            if voiced and np.isfinite(freq):
                midi_track.append(self.hz_to_midi(freq))
            else:
                midi_track.append(None)

        midi_track = np.array(midi_track, dtype=object)

        slices = np.array_split(midi_track, 4)

        results = []

        for slice_midis in slices:

            self.spectrogram = np.roll(self.spectrogram, -1, axis=1)
            self.spectrogram[:, -1] = 0

            valid = [m for m in slice_midis if m is not None]

            if len(valid) == 0:
                results.append("no pitch")
                continue

            midi = collections.Counter(valid).most_common(1)[0][0]

            if self.MIDI_MIN <= midi <= self.MIDI_MAX:
                row = self.MIDI_MAX - midi
                self.spectrogram[row, -1] = 1

            results.append(self.midi_to_note_name(midi))

        return results

    # =============================
    # STREAM CONTROL
    # =============================
    def start(self):
        self.stream.start()

    def stop(self):
        self.stream.stop()
        self.stream.close()