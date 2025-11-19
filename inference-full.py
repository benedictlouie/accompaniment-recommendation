import torch
import numpy as np
from full import ChordTransformer
from constants import DEVICE, NUM_CLASSES, MELODY_NOTES_PER_BEAT, CHORD_CLASSES
from prepare_training_data_smoother import break_down_one_song_into_sequences
from plot_chords import plot_chords_over_time
from play import npz_to_midi

def generate_chords(model, melody, max_len=32):
    """
    melody: np.array or torch tensor (1, T, feature_dim)
    returns: predicted chord indices (1, T)
    """
    model.eval()
    if isinstance(melody, np.ndarray):
        melody = torch.tensor(melody, dtype=torch.float32)
    melody = melody.to(DEVICE)
    B, T, _ = melody.shape

    with torch.no_grad():
        outputs = model(melody, chords=None, teacher_forcing_ratio=0.0)  # fully autoregressive
        preds = outputs.argmax(dim=-1)  # (B,T)
    return preds.cpu().numpy()

if __name__ == "__main__":
    model = ChordTransformer().to(DEVICE)
    checkpoint = torch.load("checkpoints/full.pth", map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    # example melody
    song_num = 676
    melody, target_chords = break_down_one_song_into_sequences(song_num, test=True)
    predicted_chords = generate_chords(model, melody)

    target_chords = target_chords[:, -1].flatten().tolist()
    target_chords = [CHORD_CLASSES[tgt] for tgt in target_chords]
    
    predicted_chords = predicted_chords[:, -1].flatten().tolist()
    predicted_chords = [CHORD_CLASSES[pred] for pred in predicted_chords]

    plot_chords_over_time(predicted_chords, target_chords)
    npz_to_midi(song_num, predicted_chords)