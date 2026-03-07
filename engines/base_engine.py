from abc import ABC, abstractmethod
import numpy as np
from utils.constants import STEPS_PER_BEAT, BEATS_PER_BAR, NOTE_TO_MIDI, INPUT_DIM, MEMORY


class BaseChordEngine(ABC):

    def __init__(self, tempo):
        self.set_tempo(tempo)
        self.last_bar = None
        self.history = np.zeros((0, INPUT_DIM))

    def set_tempo(self, tempo):
        self.beat_duration = 60 / tempo
        self.step_duration = self.beat_duration / STEPS_PER_BEAT

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def process_beat(self, notes_played, beat_start_time, beat_index):
        """
        Called once per beat.

        Returns:
            chord_name (str or None),
            duration_seconds (float)
        """
        pass
    
    def build_history(self, notes_played, beat_start, beat_index):

        grid = -np.ones(STEPS_PER_BEAT)

        for step in range(STEPS_PER_BEAT):
            step_start = beat_start + step * self.step_duration
            step_end = step_start + self.step_duration

            active_midis = []

            for note, start, end in notes_played:
                overlap_start = max(start, step_start)
                overlap_end = min(end, step_end)

                if overlap_end > overlap_start:
                    active_midis.append(NOTE_TO_MIDI[note])

            if active_midis:
                grid[step] = np.max(active_midis)

        strong_flag = 1 if beat_index == 1 else 0
        grid = np.concatenate(([strong_flag], grid))

        new_beat = grid.reshape(1, INPUT_DIM)

        if len(self.history) == 0:
            padding = np.tile(self._empty_beat(), MEMORY)
            padding = padding.reshape(MEMORY, INPUT_DIM)
            history = padding
        else:
            history = self.history

        history = np.vstack((history, new_beat))[-MEMORY:]
        self.last_bar = history[-BEATS_PER_BAR:, -STEPS_PER_BEAT:].flatten()
        return history
    
    def _empty_beat(self):
        grid = -np.ones(INPUT_DIM)
        grid[0] = 0
        return grid
    