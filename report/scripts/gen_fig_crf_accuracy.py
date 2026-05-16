"""
Retrain SmallChordClassifier and save train/val top-1 accuracy curve.
Run from repo root:
    python report/scripts/gen_fig_crf_accuracy.py

Saves: report/figures/crf_accuracy.png
Does NOT overwrite the existing model checkpoint.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import torch
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from CRF.chord_melody_relation import SmallChordClassifier, melody_histogram
from CRF.chord_transition_prior import get_bar_chords
from data.prepare_training_data import simplify_chord
from utils.FifthsCircleLoss import FifthsCircleLoss
from utils.constants import NUM_CLASSES, REVERSE_CHORD_MAP

OUT_PATH = "report/figures/crf_accuracy.png"
EPOCHS   = 20
BATCH    = 128
LR       = 1e-3
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# ── Load data (same splits as original training) ──────────────────────────────
print("Loading training data (songs 1-800, 12 transpositions)...")
train_inputs, train_targets = np.zeros((0, 13)), []
for song_num in range(1, 801):
    path = f"data/pop/melody_chords/{song_num:03d}.npz"
    if not os.path.exists(path): continue
    data = np.load(path, allow_pickle=True)
    sb, mel = data['strong_beats'], data['melody']
    chords  = get_bar_chords(sb, data['chords'])
    for t in range(-6, 6):
        bars = melody_histogram(sb, np.where(mel > 10, mel + t, -1))
        train_inputs = np.vstack((train_inputs, bars))
        for i in range(len(bars)):
            if bars[i][0] == 1:
                train_targets.append(NUM_CLASSES - 1)
                continue
            idx = REVERSE_CHORD_MAP.get(simplify_chord(chords[i]), NUM_CLASSES - 1)
            if idx != NUM_CLASSES - 1:
                idx = (idx + 2 * t) % (NUM_CLASSES - 1)
            train_targets.append(idx)
    if song_num % 100 == 0:
        print(f"  {song_num}/800...")

print("Loading val data (songs 801-909)...")
val_inputs, val_targets = np.zeros((0, 13)), []
for song_num in range(801, 910):
    path = f"data/pop/melody_chords/{song_num:03d}.npz"
    if not os.path.exists(path): continue
    data = np.load(path, allow_pickle=True)
    sb, mel = data['strong_beats'], data['melody']
    chords  = get_bar_chords(sb, data['chords'])
    bars    = melody_histogram(sb, mel)
    val_inputs = np.vstack((val_inputs, bars))
    val_targets += [REVERSE_CHORD_MAP.get(simplify_chord(chords[i]), NUM_CLASSES - 1)
                    for i in range(len(bars))]

train_targets = np.array(train_targets)
val_targets   = np.array(val_targets)
print(f"Train bars: {len(train_inputs)}, Test bars: {len(val_inputs)}")

# Non-augmented train eval set (songs 1-800, no transposition) — matches eval_models.py
print("Loading non-augmented train eval data (songs 1-800, no transposition)...")
eval_tr_inputs, eval_tr_targets = np.zeros((0, 13)), []
for song_num in range(1, 801):
    path = f"data/pop/melody_chords/{song_num:03d}.npz"
    if not os.path.exists(path): continue
    data = np.load(path, allow_pickle=True)
    sb, mel = data['strong_beats'], data['melody']
    chords  = get_bar_chords(sb, data['chords'])
    bars    = melody_histogram(sb, mel)
    eval_tr_inputs = np.vstack((eval_tr_inputs, bars))
    eval_tr_targets += [REVERSE_CHORD_MAP.get(simplify_chord(chords[i]), NUM_CLASSES - 1)
                        for i in range(len(bars))]
eval_tr_targets = np.array(eval_tr_targets)
print(f"Train eval bars (no aug): {len(eval_tr_inputs)}")

# ── Build data loaders ────────────────────────────────────────────────────────
X_tr = torch.tensor(train_inputs,    dtype=torch.float32)
y_tr = torch.tensor(train_targets,   dtype=torch.long)
X_et = torch.tensor(eval_tr_inputs,  dtype=torch.float32)
y_et = torch.tensor(eval_tr_targets, dtype=torch.long)
X_v  = torch.tensor(val_inputs,      dtype=torch.float32)
y_v  = torch.tensor(val_targets,     dtype=torch.long)

train_loader    = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_tr, y_tr), batch_size=BATCH, shuffle=True)
eval_tr_loader  = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_et, y_et), batch_size=BATCH, shuffle=False)
test_loader     = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_v,  y_v),  batch_size=BATCH, shuffle=False)

def eval_accuracy(loader, n_total):
    correct = 0
    with torch.no_grad():
        for xb, yb in loader:
            preds    = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
    return correct / n_total * 100.0

# ── Train (fresh weights) ─────────────────────────────────────────────────────
model     = SmallChordClassifier(NUM_CLASSES)
criterion = FifthsCircleLoss().cpu()
optimizer = optim.Adam(model.parameters(), lr=LR)

train_accs, test_accs = [], []

# Epoch 0: evaluate before any training
model.eval()
train_accs.append(eval_accuracy(eval_tr_loader, len(X_et)))
test_accs.append(eval_accuracy(test_loader,     len(X_v)))
print(f"  Epoch 0/{EPOCHS}  train_acc={train_accs[-1]:.2f}%  test_acc={test_accs[-1]:.2f}%")

print(f"\nTraining for {EPOCHS} epochs...")
for epoch in range(EPOCHS):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        criterion(model(xb), yb).backward()
        optimizer.step()

    model.eval()
    train_accs.append(eval_accuracy(eval_tr_loader, len(X_et)))
    test_accs.append(eval_accuracy(test_loader,     len(X_v)))
    print(f"  Epoch {epoch+1}/{EPOCHS}  train_acc={train_accs[-1]:.2f}%  test_acc={test_accs[-1]:.2f}%")

# ── Plot ──────────────────────────────────────────────────────────────────────
epochs = range(0, EPOCHS + 1)
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(epochs, train_accs, color='steelblue', linewidth=1.5, label='Train accuracy')
ax.plot(epochs, test_accs,  color='firebrick', linewidth=1.5, label='Test accuracy')
ax.set_xlabel("Epoch")
ax.set_ylabel("Top-1 Accuracy (%)")
ax.legend(frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
print(f"\nSaved → {OUT_PATH}")
