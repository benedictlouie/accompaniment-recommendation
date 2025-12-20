import torch 
import numpy as np
from CRF.chord_transition_prior import get_bar_chords, simplify_chord
from CRF.chord_melody_relation import predict_chords, melody_histogram
from utils.constants import FIFTHS_CHORD_LIST, FIFTHS_CHORD_INDICES, CHORD_CLASSES, NUM_CLASSES, REVERSE_CHORD_MAP, MAJOR, MINOR, TEMPERATURE
from utils.plot_chords import plot_chords_over_time
from utils.play import npz_to_midi
from utils.FifthsCircleLoss import FifthsCircleLoss


def compute_fifths_circle_loss(pred_chord, true_chord):
    fifthsCircleLoss = FifthsCircleLoss()
    predicted_class = torch.tensor(REVERSE_CHORD_MAP[pred_chord])
    pred_coords = fifthsCircleLoss.map_to_circle(predicted_class).unsqueeze(0)
    actual_class = torch.tensor(REVERSE_CHORD_MAP[true_chord])
    target_coords = fifthsCircleLoss.map_to_circle(actual_class).unsqueeze(0)
    loss = torch.norm(pred_coords - target_coords, dim=1).mean()
    return loss


def key_probs(X):
    scores = [0 for _ in range(24)]
    for row in X:
        pc = row[1:13].astype(float)
        if pc.sum() > 0: pc /= pc.sum()
        else: continue
        for k in range(12):
            prof = np.roll(MAJOR, k)
            scores[2*k] += (np.corrcoef(pc, prof)[0, 1])
            prof = np.roll(MINOR, k)
            scores[2*k+1] += (np.corrcoef(pc, prof)[0, 1])

    s = np.array(scores)
    s = np.nan_to_num(s)  # handle degenerate rows
    e = np.exp(s - np.max(s))
    probs = e / e.sum()
    return probs


if __name__ == "__main__":

    song_num = 867
    song_num_str = f"{song_num:03d}"

    npz_path = f'data/pop/melody_chords/{song_num_str}.npz'
    data = np.load(npz_path, allow_pickle=True)
    strong_beats = data["strong_beats"]
    melody = data["melody"]
    chords = get_bar_chords(strong_beats, data["chords"])

    bars = melody_histogram(strong_beats, melody)

    probs, _ = predict_chords(bars) # (N, 25)
    probs = np.array(probs)
    
    num_steps, num_chords = probs.shape
    log_probs = np.log(probs + 1e-12)

    transition_matrix = np.load("chord_transition_matrix.npy") # (25, 25)
    transitions = np.sum(transition_matrix, axis=2) + 1e-12
    transitions /= transitions.sum(axis=1)
    log_transitions = np.log(transitions) * 0.3

    # Initialize delta table
    delta = np.zeros((num_steps, num_chords))
    delta[0] = log_probs[0]  # first timestep

    # Real key
    with open(f'data/pop/POP909/{song_num_str}/key_audio.txt', 'r') as file:
        key = file.readline().strip().split('\t')[2]
        key = simplify_chord(key)
    print(f"Real key: {key}")

    # Recursion: delta[t, j] = max_k (delta[t-1, k] + log_transitions[k, j]) + log_probs[t, j]
    for t in range(1, num_steps):

        # key_prob = np.zeros(NUM_CLASSES-1)
        # key_prob[FIFTHS_CHORD_INDICES[key]] = 1

        key_prob = key_probs(bars[max(0,t-16):t, :])
        rearrange = np.array([FIFTHS_CHORD_INDICES[CHORD_CLASSES[i]]-1 for i in range(NUM_CLASSES-1)])
        key_prob = key_prob[np.argsort(rearrange)]

        probs2 = np.sum(transition_matrix, axis=1) * key_prob
        probs2 = np.sum(probs2, axis=1).flatten() + 1e-12
        probs2 /= np.sum(probs2)
        log_probs2 = np.log(probs2)

        delta[t] = np.max(delta[t-1][:, None] + log_probs[t-1] + log_probs2 + log_transitions, axis=0)
    
    probs_temp = torch.softmax(torch.tensor(delta) / TEMPERATURE, dim=1)
    predicted_destinations = torch.multinomial(probs_temp, num_samples=1).squeeze(1)
    
    predicted = [FIFTHS_CHORD_LIST[pred] for pred in predicted_destinations]
    plot_chords_over_time(predicted, chords)    

    losses = [compute_fifths_circle_loss(pred_chord, true_chord) for pred_chord, true_chord in zip(predicted, chords)]
    print("Average Loss:", sum(losses)/len(losses))

    npz_to_midi(song_num, [pred for pred in predicted for _ in range(4)])
    