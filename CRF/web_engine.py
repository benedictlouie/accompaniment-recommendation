"""
Numpy-only CRF chord engine for web/serverless deployment.
No torch — weights are pre-exported to deploy/crf_nn.npz.
"""
import numpy as np
import os

_DEPLOY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "deploy")

_weights          = None
_transition_matrix = None


def _load_weights():
    global _weights
    if _weights is None:
        data = np.load(os.path.join(_DEPLOY, "crf_nn.npz"))
        _weights = {k: data[k] for k in data.files}
    return _weights


def _load_transition():
    global _transition_matrix
    if _transition_matrix is None:
        _transition_matrix = np.load(os.path.join(_DEPLOY, "crf_transition.npy"))
    return _transition_matrix


# ── Lazy imports of pure-numpy constants (no torch path triggered) ─────────────

def _constants():
    from utils.constants import (
        NUM_CLASSES, FIFTHS_CHORD_INDICES, CHORD_CLASSES,
        FIFTHS_CHORD_LIST, TEMPERATURE, MAJOR, MINOR
    )
    return NUM_CLASSES, FIFTHS_CHORD_INDICES, CHORD_CLASSES, FIFTHS_CHORD_LIST, TEMPERATURE, MAJOR, MINOR


# ── MLP forward pass ───────────────────────────────────────────────────────────

def _mlp_forward(x, w):
    """4-layer MLP matching SmallChordClassifier architecture."""
    x = x @ w['net.0.weight'].T + w['net.0.bias']
    x = np.maximum(0, x)
    x = x @ w['net.2.weight'].T + w['net.2.bias']
    x = np.maximum(0, x)
    # dropout skipped at inference
    x = x @ w['net.5.weight'].T + w['net.5.bias']
    x = np.maximum(0, x)
    x = x @ w['net.7.weight'].T + w['net.7.bias']
    return x


def _softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _predict_bar(bars: np.ndarray) -> np.ndarray:
    """
    bars: (13,) pitch-class histogram (bin 0 = silence, bins 1-12 = C..B).
    Returns (NUM_CLASSES,) probability vector in FIFTHS_CHORD_LIST order.
    """
    NUM_CLASSES, FIFTHS_CHORD_INDICES, CHORD_CLASSES, _, TEMPERATURE, _, _ = _constants()
    w = _load_weights()
    logits = _mlp_forward(bars[np.newaxis].astype(np.float32), w)[0]  # (NUM_CLASSES,)

    rearrange = np.array([FIFTHS_CHORD_INDICES[CHORD_CLASSES[i]] - 1 for i in range(NUM_CLASSES)])
    probs = _softmax(logits / TEMPERATURE)
    return probs[np.argsort(rearrange)]  # reorder to FIFTHS_CHORD_LIST order


def _key_probs(bar_history: np.ndarray) -> np.ndarray:
    """
    bar_history: (N, 13).  Returns (24,) key probabilities in chromatic order
    (same layout as CHORD_CLASSES without N).
    """
    _, _, _, _, _, MAJOR, MINOR = _constants()
    scores = np.zeros(24, dtype=float)
    for row in bar_history:
        pc = row[1:13].astype(float)
        if pc.sum() > 0:
            pc /= pc.sum()
        else:
            continue
        for k in range(12):
            scores[2*k]   += np.corrcoef(pc, np.roll(MAJOR, k))[0, 1]
            scores[2*k+1] += np.corrcoef(pc, np.roll(MINOR, k))[0, 1]
    scores = np.nan_to_num(scores)
    e = np.exp(scores - scores.max())
    return e / e.sum()


# ── Stateless Viterbi step ─────────────────────────────────────────────────────

def step(bars: np.ndarray,
         delta: np.ndarray | None,
         bar_history: np.ndarray) -> tuple[str | None, np.ndarray, np.ndarray]:
    """
    One CRF bar step.

    Args:
        bars:        (13,) pitch-class histogram for this bar.
        delta:       (NUM_CLASSES,) Viterbi accumulator, or None on first bar.
        bar_history: (N, 13) previous bar histograms (up to 8 kept).

    Returns:
        chord:           predicted chord string, or None if silence.
        new_delta:       updated (NUM_CLASSES,) accumulator.
        new_bar_history: updated (N+1, 13) history (trimmed to 8 rows).
    """
    NUM_CLASSES, FIFTHS_CHORD_INDICES, CHORD_CLASSES, FIFTHS_CHORD_LIST, _, _, _ = _constants()
    transition_matrix = _load_transition()  # (25, 25, 24)

    # Append bar to history
    new_history = np.vstack([bar_history, bars]) if len(bar_history) else bars[np.newaxis]
    if len(new_history) > 8:
        new_history = new_history[-8:]

    log_probs = np.log(_predict_bar(bars) + 1e-12)  # (NUM_CLASSES,)

    # Marginal transition matrix (sum over key dimension)
    transitions = np.sum(transition_matrix, axis=2) + 1e-12
    transitions /= transitions.sum(axis=1, keepdims=True)
    log_transitions = np.log(transitions) * 0.3  # (NUM_CLASSES, NUM_CLASSES)

    # Key-conditioned transition prior
    key_prob = _key_probs(new_history)  # (24,) in chromatic/CHORD_CLASSES order
    rearrange = np.array([FIFTHS_CHORD_INDICES[CHORD_CLASSES[i]] - 1 for i in range(NUM_CLASSES - 1)])
    key_prob_fifths = key_prob[np.argsort(rearrange)]  # (NUM_CLASSES-1,) in fifths order

    probs2 = np.sum(transition_matrix, axis=1) * key_prob_fifths  # (NUM_CLASSES, NUM_CLASSES-1)
    probs2 = np.sum(probs2, axis=1) + 1e-12                        # (NUM_CLASSES,)
    probs2 /= probs2.sum()
    log_probs2 = np.log(probs2)

    if delta is None:
        # First bar: initialise with emissions + key prior (no transition yet)
        new_delta = log_probs + log_probs2
    else:
        # Proper Viterbi (matches crf.py line 89):
        # combined[k,j] = delta[k] + log_transitions[k,j]  → max over k (incoming state)
        # then add emissions for j
        # delta[:, None] = (NC, 1) so combined[k,j] = delta[k] + log_transitions[k,j]
        combined = delta[:, None] + log_transitions          # (NC, NC)
        new_delta = np.max(combined, axis=0) + log_probs + log_probs2

    decision = FIFTHS_CHORD_LIST[np.argmax(new_delta)]
    chord = None if decision == 'N' else decision

    return chord, new_delta.astype(np.float32), new_history.astype(np.float32)
