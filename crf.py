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

    song_num = 676
    song_num_str = f"{song_num:03d}"

    npz_path = f'pop/melody_chords/{song_num_str}.npz'
    data = np.load(npz_path, allow_pickle=True)
    strong_beats = data["strong_beats"]
    melody = data["melody"]
    chords = get_bar_chords(strong_beats, data["chords"])

    bars = melody_histogram(strong_beats, melody)
    probs, _ = predict_chords(bars) # (N, 25)
    transition_matrix = np.load("chord_transition_matrix.npy") # (25, 25)

    predicted = []
    current_chord_index = 0
    for prob in probs:
        predicted.append(FIFTHS_CHORD_LIST[current_chord_index])
        scores = prob * transition_matrix[current_chord_index] ** 0.3
        next_chord_index = np.argmax(scores).item()
        current_chord_index = next_chord_index
    
    plot_chords_over_time(predicted, chords)    

    losses = [compute_fifths_circle_loss(pred_chord, true_chord) for pred_chord, true_chord in zip(predicted, chords)]
    print("Average Loss:", sum(losses)/len(losses))

    npz_to_midi(song_num, [pred for pred in predicted for _ in range(4)])
    