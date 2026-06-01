import queue
import threading
import torch
import numpy as np
from AR.ar_transformer import OUTPUT_DIM, TransformerModel,load_model_checkpoint
from CRF.crf import compute_fifths_circle_loss
from data.prepare_training_data import break_down_one_song_into_sequences
from utils.constants import DEVICE, CHORD_CLASSES_ALL, REVERSE_CHORD_MAP, MEMORY, STEPS_PER_BEAT, NUM_CLASSES_ALL, DEVICE, CHORD_TO_TETRAD, INPUT_DIM, CHORD_EMBEDDING_LENGTH, TEMPERATURE
from utils.FifthsCircleLoss import FifthsCircleLoss
from utils.plot_chords import plot_chords_over_time
from utils.play import npz_to_midi

_log_queue: queue.Queue = queue.Queue()

def _log_worker():
    while True:
        msg = _log_queue.get()
        if msg is None:
            break
        print(msg)
        _log_queue.task_done()

_log_thread = threading.Thread(target=_log_worker, daemon=True)
_log_thread.start()

def _log(msg):
    _log_queue.put(msg)

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
    _log(str(melody[:, -1].cpu().numpy()))

    # melody[(melody[:, :, -STEPS_PER_BEAT:] == -1).all(dim=2).all(dim=1)] = -1

    with torch.inference_mode():
        outputs = model(melody)  # fully autoregressive

        probs = torch.nn.functional.softmax(outputs / TEMPERATURE, dim=-1)

        last_probs = probs[:, -1, :]          # (B, V)
        top_probs, top_indices = torch.topk(last_probs, k=8, dim=-1)
        for prob, index in zip(top_probs, top_indices):
            _log(str([str(CHORD_CLASSES_ALL[chord_idx.item()]) for chord_idx in index]))

        preds = torch.multinomial(
            probs.view(-1, probs.size(-1)), 1
        ).view(outputs.size(0), outputs.size(1))
        # preds = outputs.argmax(dim=-1)  # (B,T)

    if target is not None:
        if isinstance(target, np.ndarray):
            target = torch.tensor(target, dtype=torch.long)

        target = target.to(DEVICE)
        criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        loss = criterion(outputs[:, -1, :], target[:, -1].squeeze(-1))
        _log(f"Final-step Average Cross Entropy: {loss.item()}")

    for i in range(len(preds)):
        if target is not None:
            _log("Actual sequence:\t" + str([str(CHORD_CLASSES_ALL[chord_idx.item()]) for chord_idx in target[i, -10:].squeeze(-1)]))
        _log("Predicted sequence:\t" + str([str(CHORD_CLASSES_ALL[chord_idx.item()]) for chord_idx in preds[i, -10:]]))

    _log(str(preds[:, -1].cpu().numpy()))
    return preds.cpu().numpy()

if __name__ == "__main__":
    model = TransformerModel(INPUT_DIM, NUM_CLASSES_ALL).to(DEVICE)
    checkpoint = torch.load("checkpoints/transformer_model.pth", map_location=DEVICE)
    model.load_state_dict(checkpoint)

    # example melody
    song_num = 707
    npz_path = f"data/pop/melody_chords/{song_num:03d}.npz"
    melody, target_chords = break_down_one_song_into_sequences(npz_path, test=True)
    predicted_chords = generate_chords(model, melody, target_chords)

    target_chords = target_chords[:, -1].flatten().tolist()
    target_chords = [CHORD_CLASSES_ALL[tgt] for tgt in target_chords]
    
    predicted_chords = predicted_chords[:, -1].flatten().tolist()
    predicted_chords = [CHORD_CLASSES_ALL[pred] for pred in predicted_chords]

    # plot_chords_over_time(predicted_chords, target_chords)

    npz_to_midi(song_num, predicted_chords)
