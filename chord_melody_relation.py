import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns

from constants import ROOTS, CHORD_CLASSES, NUM_CLASSES, REVERSE_CHORD_MAP, FIFTHS_CHORD_LIST, FIFTHS_CHORD_INDICES, TEMPERATURE
from prepare_training_data import simplify_chord
from chord_transition_prior import get_bar_chords
from FifthsCircleLoss import FifthsCircleLoss

def melody_histogram(trig, mel, plot=False):
    """
    For EACH segment between triggers:
        bin 0  = count of values == -1
        bins 1–12 = (value % 12)

    Returns:
        hist_list : list of 13-bin normalized histograms
        segments  : list of flattened melody segments
    """

    trig = trig.squeeze()
    start_idx = np.where(trig == 1)[0]

    segments = []
    for i in range(len(start_idx)):
        s = start_idx[i]
        e = start_idx[i+1] if i+1 < len(start_idx) else len(mel)
        segment = mel[s:e].reshape(-1)

        fill_value = None
        for i, x in enumerate(segment):
            if x != "N":
                fill_value = x
            elif fill_value is not None:
                segment[i] = fill_value

        segments.append(segment)
    
    bars = []

    for seg_i, pooled in enumerate(segments):
        bins = np.zeros(13, dtype=float)
        L = len(pooled)

        for i in range(L):
            val = pooled[i]
            w = 1
            if val == -1:
                continue
            else:
                m = int(val % 12)
                bins[m + 1] += w

        total = bins.sum()
        if total > 0:
            bins /= total
        else:
            bins[0] = 1

        if plot:
            labels = ["N"] + list(ROOTS)
            plt.figure(figsize=(10, 4))
            plt.bar(range(13), bins, tick_label=labels, color="steelblue", alpha=0.8)
            plt.xlabel("Bin")
            plt.ylabel("Normalised score")
            plt.title(f"Melody Histogram of this bar {seg_i}")
            plt.tight_layout()
            plt.show()

        bars.append(bins)

    return np.array(bars)


class SmallChordClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(13, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)
    

def train_model(
        X_train, y_train,
        X_val, y_val,
        num_classes,
        model_path="checkpoints/small_melody_chord_model.pth",
        epochs=20,
        batch_size=128,
        lr=1e-3
    ):

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train.squeeze(), dtype=torch.long)

    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val.squeeze(), dtype=torch.long)

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model
    model = SmallChordClassifier(num_classes)

    if os.path.exists(model_path):
        print("Loading existing model...")
        model.load_state_dict(torch.load(model_path))

    criterion = FifthsCircleLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        # ----- Training -----
        model.train()
        total_train_loss = 0.0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * xb.size(0)

        avg_train_loss = total_train_loss / len(train_dataset)

        # ----- Validation -----
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                loss = criterion(logits, yb)
                total_val_loss += loss.item() * xb.size(0)

        avg_val_loss = total_val_loss / len(val_dataset)

        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {avg_train_loss:.4f} - "
              f"Val Loss: {avg_val_loss:.4f}")

        # Save model every epoch
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)

    return model


def predict_chords(X, y_true=None, model_path="checkpoints/small_melody_chord_model.pth", num_classes=NUM_CLASSES, plot_cm=False):
    """
    X: (N,13) numpy array of histogram inputs
    y_true: (N,1) or (N,) ground truth class indices (optional)
    plot_cm: if True, plots the confusion matrix
    
    Returns:
        preds: predicted class indices (N,)
        accuracy: float (0-1) if y_true is provided, else None
    """
    # Load model
    model = SmallChordClassifier(num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        logits = model(X_tensor)
        probs = nn.functional.softmax(logits, dim=1)
        max_probs, preds = torch.max(probs, dim=1)

        rearrange = np.array([FIFTHS_CHORD_INDICES[CHORD_CLASSES[i]]-1 for i in range(NUM_CLASSES)])
        probs = probs[:, np.argsort(rearrange)]
    
        probs_temp = torch.softmax(torch.tensor(logits) / TEMPERATURE, dim=1)
        preds = torch.multinomial(probs_temp, num_samples=1).squeeze(1)
        preds = preds.numpy()

    accuracy = None

    if y_true is not None:
        y_true = np.array(y_true).squeeze()
        y_true = np.array([FIFTHS_CHORD_INDICES[CHORD_CLASSES[y]] for y in y_true])
        preds = np.array([FIFTHS_CHORD_INDICES[CHORD_CLASSES[y]] for y in preds])
        
        mask = preds > 0
        y_true = y_true[mask]
        preds = preds[mask]
        accuracy = np.mean(preds == y_true)
        
        if plot_cm:
            cm = confusion_matrix(y_true, preds, labels=np.arange(num_classes))
            plt.figure(figsize=(8,6))
            sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', cbar=True, xticklabels=FIFTHS_CHORD_LIST, yticklabels=FIFTHS_CHORD_LIST)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"Confusion Matrix - Accuracy: {accuracy*100:.2f}%")
            plt.show()

    return probs, accuracy


if __name__ == "__main__":

    train_inputs = np.ndarray((0, 13))
    train_targets = []
    for song_num in range(1, 801):
        song_num_str = f"{song_num:03d}"
        npz_path = f'pop/melody_chords/{song_num_str}.npz'
        data = np.load(npz_path, allow_pickle=True)
        strong_beats = data['strong_beats']
        melody = data["melody"]
        chords = get_bar_chords(strong_beats, data["chords"])

        for transpose in range(-6,6):
            bars = melody_histogram(strong_beats, np.where(melody > 10, melody+transpose, -1))
            train_inputs = np.vstack((train_inputs, bars))
            for i in range(len(bars)):
                
                # No melody no chord
                if bars[i][0] == 1:
                    train_targets.append(NUM_CLASSES-1)
                    continue
                
                chord_index = REVERSE_CHORD_MAP.get(simplify_chord(chords[i]), NUM_CLASSES-1)
                if chord_index != NUM_CLASSES-1:
                    chord_index += 2 * transpose
                    chord_index %= (NUM_CLASSES-1)
                train_targets.append(chord_index)

    train_targets = np.array(train_targets)


    val_inputs = np.ndarray((0,13))
    val_targets = []
    for song_num in range(801, 910):
        song_num_str = f"{song_num:03d}"
        npz_path = f'pop/melody_chords/{song_num_str}.npz'
        data = np.load(npz_path, allow_pickle=True)
        strong_beats = data['strong_beats']
        melody = data["melody"]
        chords = get_bar_chords(strong_beats, data["chords"])
        bars = melody_histogram(strong_beats, melody)
        val_inputs = np.vstack((val_inputs, bars))
        val_targets += [REVERSE_CHORD_MAP.get(simplify_chord(chords[i]), NUM_CLASSES-1) for i in range(len(bars))]
    val_targets = np.array(val_targets)

    # train_model(train_inputs, train_targets, val_inputs, val_targets,
    #             num_classes=NUM_CLASSES, model_path="checkpoints/small_melody_chord_model.pth", epochs=20)

    # Inference
    predict_chords(val_inputs, val_targets, plot_cm=True)