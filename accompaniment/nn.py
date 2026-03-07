import numpy as np
from data.lpd.extract import get_groove
from utils.constants import TEMPERATURE, NN, DRUMS, PIANOS, GUITARS, BASSES, STEPS_PER_BAR
from scipy.special import softmax

def find_similar_loops(melody, top_k=10):
    """
    Returns nearest neighbour indices + distances for the melody groove.
    """
    query = get_groove(melody)
    assert len(query) == STEPS_PER_BAR + 1
    dist, idx = NN.kneighbors(query.reshape(1, -1), n_neighbors=top_k)
    return idx[0], dist[0]


def sample_from_candidates(candidates, dist):
    """
    Temperature sampling over candidates.
    """

    logits = -dist / TEMPERATURE
    probs = softmax(logits)
    choice = np.random.choice(len(candidates), p=probs)
    return candidates[choice]


def get_all_loops(melody, top_k=10):
    """
    Retrieve a coherent multi-instrument loop.

    Returns:
        drum_loop
        piano_loop
        guitar_loop
        bass_loop
    """

    idx, dist = find_similar_loops(melody, top_k)

    drum_candidates = DRUMS[idx]
    piano_candidates = PIANOS[idx]
    guitar_candidates = GUITARS[idx]
    bass_candidates = BASSES[idx]

    logits = -dist / TEMPERATURE
    probs = softmax(logits)

    choice = np.random.choice(len(idx), p=probs)

    return (
        drum_candidates[choice],
        piano_candidates[choice],
        guitar_candidates[choice],
        bass_candidates[choice],
    )