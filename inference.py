import torch
import numpy as np
# from model import SequenceToChordTransformer, load_checkpoint
from gru import SequenceToChordGRU, load_checkpoint
from prepare_training_data import break_down_one_song_into_sequences
from constants import DEVICE, CHORD_CLASSES, REVERSE_CHORD_MAP, MEMORY, NUM_CLASSES, DEVICE, CHORD_TO_TETRAD, INPUT_DIM, CHORD_EMBEDDING_LENGTH
from FifthsCircleLoss import FifthsCircleLoss
from plot_chords import plot_chords_over_time
from play import npz_to_midi

# ----- Load Model -----
model = SequenceToChordGRU(input_dim=INPUT_DIM)
model, _, _, _ = load_checkpoint(model, save_path='checkpoints/gru-10.pth', device=DEVICE)
model.to(DEVICE)
model.eval()

# ----- Prepare Data -----
num = 867
inputs, targets = break_down_one_song_into_sequences(num, test=True)
print("Inputs:", inputs.shape, "Targets:", targets.shape)

# Initialize last chord as "N" (no chord) index
last_chord_idx = NUM_CLASSES - 1  # assume last index is "N"
predicted_chords = [last_chord_idx]

predicted_chord_names = []
target_chord_names = []

beam_width = 22
beam_scores = None
beam_classes = None      # classes for each beam at current step

predicted_chords = []
predicted_chord_names = []
target_chord_names = []

prime = 0

for inp, target in zip(inputs, targets):
    
    # ----- Your memory update logic stays the same -----
    if prime < 0:
        prime += 1
    else:
        for i in range(-min(MEMORY, len(predicted_chord_names)), 0):
            prev_chord = CHORD_CLASSES[predicted_chords[i]]
            inp[i, -CHORD_EMBEDDING_LENGTH:] = CHORD_TO_TETRAD[prev_chord]

    # Convert to tensor
    input_tensor = torch.from_numpy(inp.astype(np.float32)).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(input_tensor)  # [1, NUM_CLASSES]
        probs = torch.softmax(logits[:, :-1], dim=1).squeeze(0).cpu().numpy()

    # ======== BEAM SEARCH OVER FULL SEQUENCE (no backtracking) ========

    logits_tensor = torch.tensor(logits, dtype=torch.float32)
    log_probs = torch.log_softmax(logits_tensor[:, :-1], dim=1).squeeze(0)  # [NUM_CLASSES]

    if beam_scores is None:
        # First time step: initialize beams
        top_log_probs, top_indices = torch.topk(log_probs, beam_width)

        beam_scores = top_log_probs.clone()
        beam_classes = top_indices.clone()

    else:
        # Expand beams → combine scores
        expanded_scores = beam_scores[:, None] + log_probs[None, :]  # [beam_width, vocab]
        flat_scores = expanded_scores.flatten()

        # Pick best beams
        top_scores, top_positions = torch.topk(flat_scores, beam_width)

        vocab = log_probs.shape[0]
        new_classes = top_positions % vocab

        beam_scores = top_scores
        beam_classes = new_classes

    # Choose final prediction for this timestep → best beam
    predicted_class = int(beam_classes[0].item())
    actual_class = int(target)

    # =====================================================

    predicted_chords.append(predicted_class)
    predicted_chord_names.append(CHORD_CLASSES[predicted_class])
    target_chord_names.append(CHORD_CLASSES[actual_class])

    # Top-N printout
    top_n = 3
    top_indices = np.argsort(probs)[-top_n:][::-1]
    for i in top_indices:
        print(f"{CHORD_CLASSES[i]}: {probs[i]:.4f}")

    print(f"Predicted: {CHORD_CLASSES[predicted_class]}, Actual: {CHORD_CLASSES[actual_class]}")

# Compute Fifths Circle Loss
def compute_fifths_circle_loss(pred_chord, true_chord):
    fifthsCircleLoss = FifthsCircleLoss()
    predicted_class = torch.tensor(REVERSE_CHORD_MAP[pred_chord])
    pred_coords = fifthsCircleLoss.map_to_circle(predicted_class).unsqueeze(0)
    actual_class = torch.tensor(REVERSE_CHORD_MAP[true_chord])
    target_coords = fifthsCircleLoss.map_to_circle(actual_class).unsqueeze(0)
    loss = torch.norm(pred_coords - target_coords, dim=1).mean()
    return loss

losses = [compute_fifths_circle_loss(pred_chord, true_chord) for pred_chord, true_chord in zip(predicted_chord_names, target_chord_names)]
print("Average Loss:", sum(losses)/len(losses))

plot_chords_over_time(predicted_chord_names, target_chord_names)
npz_to_midi(num, predicted_chord_names)