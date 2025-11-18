import torch
import numpy as np
from full import ChordTransformer
from constants import DEVICE, NUM_CLASSES, MELODY_NOTES_PER_BEAT

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
    melody = np.random.rand(1, 32, 1+MELODY_NOTES_PER_BEAT).astype(np.float32)
    chords = generate_chords(model, melody)
    print(chords)
