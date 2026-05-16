"""
Evaluate CRF and AR Transformer on the held-out validation set.
Run from the project root:  python report/scripts/eval_models.py

Metrics reported for both models:
  - Val Cross-Entropy
  - Top-1 Accuracy
  - Mean Chord2Vec PCA distance (Euclidean in PC1/PC2 of token_embeddings_32d.json)

CRF is evaluated on a random 2000-sample subset (Viterbi is slow on CPU).
Transformer is evaluated on the full val set.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.constants import (
    DEVICE, INPUT_DIM, MEMORY, NUM_CLASSES_ALL, NUM_CLASSES,
    CHORD_CLASSES, CHORD_CLASSES_ALL, REVERSE_CHORD_MAP, REVERSE_CHORD_MAP_ALL,
    FIFTHS_CHORD_LIST, FIFTHS_CHORD_INDICES, MAJOR, MINOR, QUALITY_SIMPLIFIER
)
from AR.ar_transformer import TransformerModel, MusicDataset
from data.chord2vec import load_embeddings, compute_pca_3d

VAL_NPZ    = "data/data_val.npz"
AR_CKPT    = "checkpoints/transformer_model.pth"
CRF_TRANS  = "CRF/chord_transition_matrix.npy"
EMBED_PATH = "data/token_embeddings_32d.json"
CRF_SAMPLES = 2000

# ── Shared Chord2Vec PCA reference space ─────────────────────────────────────
tokens, labels, mod, embeddings = load_embeddings(EMBED_PATH)
coords_3d = compute_pca_3d(embeddings)   # [N, 3]
coords_2d = coords_3d[:, :2]             # PC1, PC2

label_to_coord = {labels[i]: coords_2d[i] for i in range(len(labels))}

# 169-class index → 2D coord
idx_to_coord_169 = np.zeros((NUM_CLASSES_ALL, 2), dtype=np.float32)
for i, chord in enumerate(CHORD_CLASSES_ALL):
    if chord in label_to_coord:
        idx_to_coord_169[i] = label_to_coord[chord]
idx_to_coord_t = torch.tensor(idx_to_coord_169)

# 25-class index → 2D coord
idx_to_coord_25 = np.zeros((NUM_CLASSES, 2), dtype=np.float32)
for i, chord in enumerate(CHORD_CLASSES):
    if chord in label_to_coord:
        idx_to_coord_25[i] = label_to_coord[chord]


# ── AR Transformer evaluation ─────────────────────────────────────────────────
def eval_transformer():
    print("=== AR Transformer (169-class) ===")
    model = TransformerModel(INPUT_DIM, NUM_CLASSES_ALL).to(DEVICE)
    model.load_state_dict(torch.load(AR_CKPT, map_location=DEVICE))
    model.eval()
    criterion = nn.CrossEntropyLoss()

    # Random baseline for normalised dist
    rng = np.random.default_rng(0)
    ri = rng.integers(0, NUM_CLASSES_ALL, 100000)
    rj = rng.integers(0, NUM_CLASSES_ALL, 100000)
    rand_baseline = np.linalg.norm(idx_to_coord_169[ri] - idx_to_coord_169[rj], axis=1).mean()

    # Precompute simplified maj/min label for each of the 169 chord indices
    majmin_map = [simplify_to_majmin(c) for c in CHORD_CLASSES_ALL]

    def run_split(npz_path, label):
        dataset = MusicDataset(npz_path)
        loader  = DataLoader(dataset, batch_size=256, shuffle=False)
        tot_ce, tot_correct, tot_dist, tot_majmin, tot_n = 0.0, 0, 0.0, 0, 0
        n_batches = len(loader)
        print(f"  Evaluating {label} set ({n_batches} batches)...")
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(loader):
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                logits       = model(inputs)
                last_logits  = logits[:, -1, :]
                last_targets = targets[:, -1, 0]
                tot_ce      += criterion(last_logits, last_targets).item() * inputs.size(0)
                preds        = last_logits.argmax(dim=-1)
                tot_correct += (preds == last_targets).sum().item()
                pred_c       = idx_to_coord_t[preds.cpu()]
                true_c       = idx_to_coord_t[last_targets.cpu()]
                tot_dist    += torch.norm(pred_c - true_c, dim=1).sum().item()
                for p, t in zip(preds.cpu().tolist(), last_targets.cpu().tolist()):
                    tot_majmin += int(majmin_map[p] == majmin_map[t])
                tot_n       += inputs.size(0)
                if (i + 1) % 500 == 0 or (i + 1) == n_batches:
                    print(f"    {i+1}/{n_batches} batches  acc={tot_correct/tot_n*100:.1f}%", flush=True)
        n = tot_n
        return dict(ce=tot_ce/n, acc=tot_correct/n,
                    dist=tot_dist/n, dist_norm=(tot_dist/n)/rand_baseline,
                    majmin=tot_majmin/n)

    TRAIN_NPZ = "data/data_train.npz"
    tr = run_split(TRAIN_NPZ, "train")
    va = run_split(VAL_NPZ,   "val")

    print(f"\n  Train Accuracy                              : {tr['acc']*100:.2f}%")
    print(f"  Val Accuracy                                : {va['acc']*100:.2f}%")
    print(f"  Train Cross-Entropy                         : {tr['ce']:.4f}")
    print(f"  Val Cross-Entropy                           : {va['ce']:.4f}")
    print(f"  Train Major/Minor Accuracy                  : {tr['majmin']*100:.2f}%")
    print(f"  Val Major/Minor Accuracy                    : {va['majmin']*100:.2f}%")
    print(f"  Train Output-Layer PCA dist                 : {tr['dist']:.4f}")
    print(f"  Val Output-Layer PCA dist                   : {va['dist']:.4f}")
    print(f"  Train Output-Layer PCA dist (normalised)    : {tr['dist_norm']:.4f}  (rand baseline = {rand_baseline:.4f})")
    print(f"  Val Output-Layer PCA dist (normalised)      : {va['dist_norm']:.4f}  (rand baseline = {rand_baseline:.4f})")
    return tr, va


def simplify_to_majmin(chord_label):
    """Reduce any chord label to root:maj or root:min (or N)."""
    if chord_label == "N":
        return "N"
    root, qual = chord_label.split(":")
    qual_simple = QUALITY_SIMPLIFIER.get(qual, qual)
    if qual_simple not in ("maj", "min"):
        qual_simple = "maj"
    return f"{root}:{qual_simple}"


# ── CRF evaluation ────────────────────────────────────────────────────────────
def eval_crf():
    from CRF.chord_melody_relation import SmallChordClassifier, melody_histogram
    from CRF.chord_transition_prior import get_bar_chords
    from data.prepare_training_data import simplify_chord
    from utils.FifthsCircleLoss import FifthsCircleLoss

    # Load model and loss
    model = SmallChordClassifier(NUM_CLASSES)
    model.load_state_dict(torch.load("checkpoints/small_melody_chord_model.pth", map_location='cpu'))
    model.eval()

    fcl = FifthsCircleLoss(num_chords=NUM_CLASSES - 1).cpu()  # 24 maj/min chords
    chord_coords = fcl.chord_coords[:NUM_CLASSES].cpu().detach().numpy()  # [25, 3]

    # Random baseline for circle-of-fifths distance (exclude N at index 24)
    rng = np.random.default_rng(0)
    ri, rj = rng.integers(0, NUM_CLASSES - 1, 100000), rng.integers(0, NUM_CLASSES - 1, 100000)
    d_rand_fifths = np.linalg.norm(chord_coords[ri] - chord_coords[rj], axis=1).mean()
    print(f"  Circle-of-fifths random baseline: {d_rand_fifths:.4f}")

    def run_songs(song_range, label):
        tot_fifths_loss, tot_correct, tot_fifths_dist, tot_majmin, tot_n = 0.0, 0, 0.0, 0, 0
        for song_num in song_range:
            npz_path = f"data/pop/melody_chords/{song_num:03d}.npz"
            if not os.path.exists(npz_path):
                continue
            data         = np.load(npz_path, allow_pickle=True)
            strong_beats = data['strong_beats']
            melody       = data['melody']
            chords       = get_bar_chords(strong_beats, data['chords'])
            bars         = melody_histogram(strong_beats, melody)
            if len(bars) == 0:
                continue

            X = torch.tensor(bars, dtype=torch.float32)
            with torch.no_grad():
                logits = model(X)                              # [n_bars, 25]

            probs = torch.softmax(logits, dim=1).numpy()

            true_indices = []
            for b in range(len(bars)):
                true_chord = simplify_chord(chords[b]) if b < len(chords) else "N"
                true_indices.append(REVERSE_CHORD_MAP.get(true_chord, NUM_CLASSES - 1))
            true_t = torch.tensor(true_indices, dtype=torch.long)

            # FifthsCircleLoss on this song
            with torch.no_grad():
                loss_val = fcl(logits, true_t).item()
            tot_fifths_loss += loss_val * len(bars)

            for b in range(len(bars)):
                true_idx = true_indices[b]
                pred_idx = int(np.argmax(probs[b]))

                # Top-1 accuracy
                tot_correct += int(pred_idx == true_idx)

                # Circle-of-fifths distance
                dist = np.linalg.norm(chord_coords[pred_idx] - chord_coords[true_idx])
                tot_fifths_dist += dist

                # Major/minor accuracy (CRF vocab already maj/min so same as top-1 for non-N)
                pred_chord = CHORD_CLASSES[pred_idx]
                true_chord_label = CHORD_CLASSES[true_idx]
                tot_majmin += int(simplify_to_majmin(pred_chord) == simplify_to_majmin(true_chord_label))

                tot_n += 1

        return {
            "fifths_loss": tot_fifths_loss / tot_n,
            "top1_acc":    tot_correct / tot_n,
            "fifths_dist": tot_fifths_dist / tot_n,
            "fifths_dist_norm": (tot_fifths_dist / tot_n) / d_rand_fifths,
            "majmin_acc":  tot_majmin / tot_n,
            "n": tot_n,
        }

    print("\n=== CRF (25-class) ===")
    # Val: songs 801-909
    print("  Running val (songs 801-909)...")
    val = run_songs(range(801, 910), "val")

    # Train: sample 100 songs from 1-800 for speed
    train_songs = sorted(rng.choice(range(1, 801), size=100, replace=False).tolist())
    print(f"  Running train sample ({len(train_songs)} songs)...")
    trn = run_songs(train_songs, "train")

    print(f"\n  {'Metric':<35} {'Train':>10} {'Val':>10}")
    print(f"  {'-'*55}")
    print(f"  {'Top-1 Accuracy':<35} {trn['top1_acc']*100:>9.2f}% {val['top1_acc']*100:>9.2f}%")
    print(f"  {'FifthsCircleLoss':<35} {trn['fifths_loss']:>10.4f} {val['fifths_loss']:>10.4f}")
    print(f"  {'Major/Minor Accuracy':<35} {trn['majmin_acc']*100:>9.2f}% {val['majmin_acc']*100:>9.2f}%")
    print(f"  {'Mean Circle-of-Fifths dist':<35} {trn['fifths_dist']:>10.4f} {val['fifths_dist']:>10.4f}")
    print(f"  {'Mean CoF dist (normalised)':<35} {trn['fifths_dist_norm']:>10.4f} {val['fifths_dist_norm']:>10.4f}")
    print(f"  {'Random baseline (CoF dist)':<35} {d_rand_fifths:>10.4f} {d_rand_fifths:>10.4f}")

    return val


if __name__ == "__main__":
    val_crf = eval_crf()
    ar_tr, ar_va = eval_transformer()

    print("\n=== Summary ===")
    print(f"  CRF  — Top-1: {val_crf['top1_acc']*100:.2f}%  FifthsLoss: {val_crf['fifths_loss']:.4f}  Maj/Min: {val_crf['majmin_acc']*100:.2f}%  CoF dist (norm): {val_crf['fifths_dist_norm']:.4f}")
    print(f"  AR   — Train CE: {ar_tr['ce']:.4f}  Val CE: {ar_va['ce']:.4f}  Train Acc: {ar_tr['acc']*100:.2f}%  Val Acc: {ar_va['acc']*100:.2f}%  PCA dist (norm): {ar_va['dist_norm']:.4f}")
