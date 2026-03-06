import numpy as np
import torch

from engines.base_engine import BaseChordEngine
from AR.ar_transformer import TransformerModel
from AR.inference import generate_chords
from utils.constants import *

STEPS_PER_BEAT = 4                 # 16th notes
STEPS_PER_BAR = STEPS_PER_BEAT # * BEATS_PER_BAR
STEP_DURATION = BEAT_DURATION / STEPS_PER_BEAT

class ARTransformerEngine(BaseChordEngine):

    def __init__(self, checkpoint_path="checkpoints/transformer_model_feb25.pth"):
        self.device = DEVICE

        self.model = TransformerModel(INPUT_DIM, NUM_CLASSES_ALL).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()

        self.history = np.zeros((0, INPUT_DIM))

    def reset(self):
        self.history = np.zeros((0, INPUT_DIM))

    def process_beat(self, notes_played, beat_start_time, beat_index):

        padded = self.build_memory(notes_played, beat_start_time, beat_index)

        predicted = generate_chords(
            self.model,
            padded[np.newaxis, :, :]
        )[-1, -1]

        chord_name = CHORD_CLASSES_ALL[predicted]

        self.history = padded[-MEMORY:]

        return chord_name, BEAT_DURATION

    def build_memory(self, notes_played, beat_start, beat_index):

        grid = -np.ones(STEPS_PER_BEAT)

        for step in range(STEPS_PER_BEAT):
            step_start = beat_start + step * STEP_DURATION
            step_end = step_start + STEP_DURATION

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

        new_bar = grid.reshape(1, INPUT_DIM)

        if len(self.history) == 0:
            padding = np.tile(self._empty_bar(), MEMORY)
            padding = padding.reshape(MEMORY, INPUT_DIM)
            history = padding
        else:
            history = self.history

        history = np.vstack((history, new_bar))[-MEMORY:]

        return history

    def _empty_bar(self):
        grid = -np.ones(INPUT_DIM)
        grid[0] = 0
        return grid
