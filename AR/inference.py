import torch
import numpy as np
from AR.ar_transformer import OUTPUT_DIM, TransformerModel,load_model_checkpoint
from CRF.crf import compute_fifths_circle_loss
from data.prepare_training_data import break_down_one_song_into_sequences
from utils.constants import DEVICE, CHORD_CLASSES, REVERSE_CHORD_MAP, MEMORY, NUM_CLASSES, DEVICE, CHORD_TO_TETRAD, INPUT_DIM, CHORD_EMBEDDING_LENGTH
from utils.FifthsCircleLoss import FifthsCircleLoss
from utils.plot_chords import plot_chords_over_time
from utils.play import npz_to_midi

def generate_chords(model, melody, target=None):
    """
    melody: np.array or torch tensor (1, T, feature_dim)
    returns: predicted chord indices (1, T)
    """
    model.eval()
    if isinstance(melody, np.ndarray):
        melody = torch.tensor(melody, dtype=torch.float32)
    melody = melody.to(DEVICE)
    B, T, _ = melody.shape

    np.set_printoptions(threshold=np.inf)
    print(melody[:, -1].cpu().numpy())

    with torch.no_grad():
        outputs = model(melody)  # fully autoregressive
        
        temperature = 0.3
        probs = torch.nn.functional.softmax(outputs / temperature, dim=-1)
        preds = torch.multinomial(
            probs.view(-1, probs.size(-1)), 1
        ).view(outputs.size(0), outputs.size(1))

        # preds = outputs.argmax(dim=-1)  # (B,T)

    print(preds[:, -1].cpu().numpy())
    return preds.cpu().numpy()

if __name__ == "__main__":
    model = TransformerModel(INPUT_DIM, NUM_CLASSES+1).to(DEVICE)
    checkpoint = torch.load("checkpoints/transformer_model.pth", map_location=DEVICE)
    model.load_state_dict(checkpoint)

    # example melody
    song_num = 707
    npz_path = f"data/pop/melody_chords/{song_num:03d}.npz"
    melody, target_chords = break_down_one_song_into_sequences(npz_path, test=True)
    predicted_chords = generate_chords(model, melody, target_chords)

    target_chords = target_chords[:, -1].flatten().tolist()
    target_chords = [CHORD_CLASSES[tgt] for tgt in target_chords]
    
    predicted_chords = predicted_chords[:, -1].flatten().tolist()
    predicted_chords = [CHORD_CLASSES[pred] for pred in predicted_chords]

    plot_chords_over_time(predicted_chords, target_chords)

    losses = [compute_fifths_circle_loss(pred_chord, true_chord) for pred_chord, true_chord in zip(predicted_chords, target_chords)]
    print("Average Loss:", sum(losses)/len(losses))

    npz_to_midi(song_num, predicted_chords)
