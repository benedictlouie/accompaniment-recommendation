import numpy as np
from data.lpd.extract import get_groove
from utils.constants import TEMPERATURE
from scipy.special import softmax

def find_similar_drums(nn, drums, melody, top_k=10):
    query = get_groove(melody)
    assert len(query) == 17
    dist, idx = nn.kneighbors(query.reshape(1, -1), n_neighbors=top_k)
    candidates = drums[idx[0]]
    return candidates, dist[0]

def get_drum_loop(nn, drums, melody, top_k=10):
    candidates, dist = find_similar_drums(nn, drums, melody, top_k)
    logits = -dist / TEMPERATURE
    probs = softmax(logits)
    choice = np.random.choice(len(candidates), p=probs)
    return candidates[choice]