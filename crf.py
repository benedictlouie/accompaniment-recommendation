import torch 
import numpy as np
from chord_transition_prior import get_bar_chords
from chord_melody_relation import predict_chords, melody_histogram
from constants import FIFTHS_CHORD_LIST, REVERSE_CHORD_MAP
from plot_chords import plot_chords_over_time
from play import npz_to_midi
from FifthsCircleLoss import FifthsCircleLoss


def compute_fifths_circle_loss(pred_chord, true_chord):
    fifthsCircleLoss = FifthsCircleLoss()
    predicted_class = torch.tensor(REVERSE_CHORD_MAP[pred_chord])
    pred_coords = fifthsCircleLoss.map_to_circle(predicted_class).unsqueeze(0)
    actual_class = torch.tensor(REVERSE_CHORD_MAP[true_chord])
    target_coords = fifthsCircleLoss.map_to_circle(actual_class).unsqueeze(0)
    loss = torch.norm(pred_coords - target_coords, dim=1).mean()
    return loss


if __name__ == "__main__":

    song_num = 867
    song_num_str = f"{song_num:03d}"

    npz_path = f'pop/melody_chords/{song_num_str}.npz'
    data = np.load(npz_path, allow_pickle=True)
    strong_beats = data["strong_beats"]
    melody = data["melody"]
    chords = get_bar_chords(strong_beats, data["chords"])

    bars = melody_histogram(strong_beats, melody)
    probs, _ = predict_chords(bars) # (N, 25)
    probs = np.array(probs)
    transition_matrix = np.load("chord_transition_matrix.npy") # (25, 25)

    num_steps, num_chords = probs.shape
    log_probs = np.log(probs + 1e-12)  # (T, C)
    log_transitions = np.log(transition_matrix + 1e-12)  # (C, C)

    # Initialize delta table
    delta = np.zeros((num_steps, num_chords))
    delta[0] = log_probs[0]  # first timestep

    # Recursion: delta[t, j] = max_k (delta[t-1, k] + log_transitions[k, j]) + log_probs[t, j]
    for t in range(1, num_steps):
        # Compute delta[t] in vectorized form: for each next chord j, compare all previous k
        # This is equivalent to t x C x C if fully expanded
        delta[t] = np.max(delta[t-1][:, None] + log_probs[t-1] + log_transitions, axis=0)
    
    temp = .5
    probs_temp = torch.softmax(torch.tensor(delta) / temp, dim=1)
    predicted_destinations = torch.multinomial(probs_temp, num_samples=1).squeeze(1)
    
    predicted = [FIFTHS_CHORD_LIST[pred] for pred in predicted_destinations]
    plot_chords_over_time(predicted, chords)    

    losses = [compute_fifths_circle_loss(pred_chord, true_chord) for pred_chord, true_chord in zip(predicted, chords)]
    print("Average Loss:", sum(losses)/len(losses))

    npz_to_midi(song_num, [pred for pred in predicted for _ in range(4)])
    