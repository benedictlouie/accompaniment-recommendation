# chord_engine.py

import numpy as np
from CRF.chord_melody_relation import predict_chords
from CRF.crf import key_probs
from utils.constants import *

class ChordEngine:
    def __init__(self):
        self.delta = np.zeros((1, NUM_CLASSES))
        self.bar_history = np.zeros((0, 13))
        self.transition_matrix = np.load("crf/chord_transition_matrix.npy")

    def reset(self):
        self.delta = np.zeros((1, NUM_CLASSES))
        self.bar_history = np.zeros((0, 13))

    def _compute_key(self):
        key_prob = key_probs(self.bar_history)

        rearrange = np.array([
            FIFTHS_CHORD_INDICES[CHORD_CLASSES[i]] - 1
            for i in range(NUM_CLASSES - 1)
        ])

        key_prob = key_prob[np.argsort(rearrange)]
        key_index = np.argmax(key_prob)
        detected_key = FIFTHS_CHORD_LIST[key_index + 1]
        print("Detected Key:", detected_key)

        return key_prob

    def _next_chord(self):
        bars = self.bar_history[-1]

        probs, _ = predict_chords([bars])
        probs = np.array(probs)

        log_probs = np.log(probs + 1e-12)

        transitions = np.sum(self.transition_matrix, axis=2) + 1e-12
        transitions /= transitions.sum(axis=1)
        log_transitions = np.log(transitions) * 0.3

        key_prob = self._compute_key()

        probs2 = np.sum(self.transition_matrix, axis=1) * key_prob
        probs2 = np.sum(probs2, axis=1).flatten() + 1e-12
        probs2 /= np.sum(probs2)

        log_probs2 = np.log(probs2)

        if len(self.bar_history) > 8:
            self.bar_history = self.bar_history[1:, :]

        combined = self.delta + log_probs + log_probs2 + log_transitions
        return np.max(combined, axis=0)

    def process_bar(self, bar_proportions):
        bars = [0 for _ in range(13)]

        for note, prop in bar_proportions.items():
            root = note[:-1]
            bars[REVERSE_ROOT_MAP[root] + 1] += prop

        if sum(bars) == 0:
            bars[0] = 1

        bars = np.array(bars) / np.sum(bars)

        self.bar_history = np.vstack((self.bar_history, bars))

        probs = self._next_chord()
        decision = FIFTHS_CHORD_LIST[np.argmax(probs)]

        if decision == 'N':
            return None

        return decision
