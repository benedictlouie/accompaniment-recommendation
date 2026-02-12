import numpy as np
from engines.base_engine import BaseChordEngine
from CRF.chord_engine import ChordEngine as InternalCRF
from utils.constants import *


class CRFChordEngine(BaseChordEngine):

    def __init__(self):
        self.engine = InternalCRF()
        self.current_bar_notes = []
        self.beat_counter = 0
        self.bar_start_time = None

    def reset(self):
        self.engine.reset()
        self.current_bar_notes = []
        self.beat_counter = 0
        self.bar_start_time = None

    def process_beat(self, notes_played, beat_start_time, beat_index):

        if self.bar_start_time is None:
            self.bar_start_time = beat_start_time

        self.current_bar_notes.extend(notes_played)
        self.beat_counter += 1

        if self.beat_counter < BEATS_PER_BAR:
            return None, 0

        # ---- BAR COMPLETE ----
        bar_end_time = beat_start_time + BEAT_DURATION

        bar_proportions = {}

        for note, start, end in self.current_bar_notes:
            start = max(start, self.bar_start_time)
            end = min(end, bar_end_time)
            duration = max(0, end - start)

            bar_proportions[note] = (
                bar_proportions.get(note, 0) + duration
            )

        total_bar_time = BEATS_PER_BAR * BEAT_DURATION

        for note in bar_proportions:
            bar_proportions[note] /= total_bar_time

        chord = self.engine.process_bar(bar_proportions)

        # Reset bar state
        self.current_bar_notes = []
        self.beat_counter = 0
        self.bar_start_time = None

        return chord, BEATS_PER_BAR * BEAT_DURATION
