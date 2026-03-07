import numpy as np
import torch

from engines.base_engine import BaseChordEngine
from AR.ar_transformer import TransformerModel
from AR.inference import generate_chords
from utils.constants import INPUT_DIM, DEVICE, NUM_CLASSES_ALL, CHORD_CLASSES_ALL, MEMORY, STEPS_PER_BEAT, NOTE_TO_MIDI

class ARTransformerEngine(BaseChordEngine):

    def __init__(self, tempo, checkpoint_path="checkpoints/transformer_model_feb25.pth"):

        super().__init__(tempo)

        self.device = DEVICE

        self.model = TransformerModel(INPUT_DIM, NUM_CLASSES_ALL).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()

    def reset(self):
        self.history = np.zeros((0, INPUT_DIM))

    def process_beat(self, notes_played, beat_start_time, beat_index):

        padded = self.build_history(notes_played, beat_start_time, beat_index)

        predicted = generate_chords(
            self.model,
            padded[np.newaxis, :, :]
        )[-1, -1]

        chord_name = CHORD_CLASSES_ALL[predicted]

        self.history = padded[-MEMORY:]

        return chord_name, self.beat_duration
    