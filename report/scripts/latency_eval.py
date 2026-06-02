"""
Latency evaluation for the AR Transformer chord model.

Simulates real-time latency where the model hasn't yet received the last N
sixteenth notes of the current (most recent) beat when it runs inference.

The input beat vector is [strong_beat_flag, n1, n2, n3, n4].
- N=0: baseline, no modification
- N=2: only n1 and n2 are known; n3 and n4 are replaced with n2
- N=3: only n1 is known; n2, n3, n4 are replaced with n1

Metric: mean output-layer embedding distance (128D Euclidean) between the
predicted chord embedding and the ground-truth chord embedding.  Uses fully
autoregressive decoding (no teacher forcing) to match the main results table.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from AR.ar_transformer import TransformerModel, MusicDataset
from utils.constants import (
    DEVICE, INPUT_DIM, MEMORY, NUM_CLASSES_ALL, CHORD_CLASSES_ALL, TEMPERATURE
)

CHECKPOINT   = "checkpoints/transformer_model.pth"
TEST_NPZ     = "data/data_val.npz"   # same split used for the main results table
BATCH_SIZE   = 50    # one batch — we only need 50 samples
NUM_SAMPLES  = 50    # random subset of the test set
RANDOM_SEED  = 42    # reproducibility


# -----------------------------------------------------------------------
# Helper: apply latency mask to a batch of inputs
# -----------------------------------------------------------------------
def apply_latency(inputs: torch.Tensor, N: int) -> torch.Tensor:
    """
    inputs: (B, MEMORY, 5)  — [strong_beat_flag, n1, n2, n3, n4]
    N: number of sixteenth-note slots NOT yet received in the final beat.

    Pitch slots in the final beat are at columns 1..4 (0-indexed).
    If N=2: last 2 slots (indices 3, 4) repeat slot at index (5 - N - 1) = index 2 (n2).
    If N=3: last 3 slots (indices 2, 3, 4) repeat slot at index 1 (n1).
    """
    x = inputs.clone()
    if N == 0:
        return x

    # The final beat is at position MEMORY-1
    # Pitch columns: 1, 2, 3, 4  (0-based within the 5-dim vector)
    # Known pitches: columns 1 .. (4 - N)  (inclusive)
    # Last known pitch column index: 1 + (4 - N) - 1 = 4 - N
    last_known_col = 4 - N          # column index in the 5-dim vector
    fill_value = x[:, -1, last_known_col].unsqueeze(1)  # (B, 1)

    for col in range(last_known_col + 1, 5):
        x[:, -1, col] = fill_value.squeeze(1)

    return x


# -----------------------------------------------------------------------
# Evaluate mean output-layer embedding distance using AR decoding
# -----------------------------------------------------------------------
def evaluate_emb_dist(model: TransformerModel, loader: DataLoader, N: int,
                      emb_weight: torch.Tensor) -> float:
    """
    Compute mean Euclidean distance (128D output-layer embedding space)
    between predicted and ground-truth chord embeddings.

    Uses fully autoregressive (AR) decoding — model(inputs_mod) with no
    target sequence — matching the main results table evaluation.

    emb_weight: (NUM_CLASSES_ALL, 128) embedding/output weight matrix on CPU.
    """
    model.eval()
    total_dist = 0.0
    total_n = 0

    with torch.inference_mode():
        for inputs, targets in loader:
            inputs  = inputs.to(DEVICE)
            targets = targets.to(DEVICE)  # (B, MEMORY+1, 1)

            inputs_mod = apply_latency(inputs, N)

            # Fully autoregressive decoding — no teacher forcing
            logits = model(inputs_mod)   # (B, MEMORY+1, NUM_CLASSES_ALL)

            # Evaluate on the LAST prediction step
            last_logits  = logits[:, -1, :]            # (B, NUM_CLASSES_ALL)
            last_targets = targets[:, -1].squeeze(-1)   # (B,)

            # Predicted chord index (argmax, deterministic)
            preds = last_logits.argmax(dim=-1)          # (B,)

            # Embedding distance in 128D output-layer weight space
            pred_emb = emb_weight[preds.cpu()]          # (B, 128)
            true_emb = emb_weight[last_targets.cpu()]   # (B, 128)
            dists = torch.norm(pred_emb - true_emb, dim=1)  # (B,)

            total_dist += dists.sum().item()
            total_n    += last_targets.size(0)

    return total_dist / total_n if total_n > 0 else 0.0


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    model = TransformerModel(INPUT_DIM, NUM_CLASSES_ALL).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.eval()
    print(f"Model loaded from {CHECKPOINT}")
    print(f"Device: {DEVICE}")

    # 128D output-layer embedding weight matrix (weight-tied with fc_out)
    # Shape: (NUM_CLASSES_ALL, 128) — stays on CPU for index lookups
    emb_weight = model.embedding_output.weight.detach().cpu()
    print(f"Output embedding matrix shape: {emb_weight.shape}")

    # ------------------------------------------------------------------
    # Load test data — 50 random samples (reproducible via RANDOM_SEED)
    # ------------------------------------------------------------------
    dataset = MusicDataset(TEST_NPZ)
    rng = np.random.default_rng(RANDOM_SEED)
    indices = rng.choice(len(dataset), size=NUM_SAMPLES, replace=False).tolist()
    subset  = Subset(dataset, indices)
    loader  = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Test set size: {len(dataset)}  |  Evaluation subset: {len(subset)} samples (seed={RANDOM_SEED})")

    # ------------------------------------------------------------------
    # Embedding distance evaluation for N=0, N=2, N=3
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("OUTPUT-LAYER EMBEDDING DISTANCE EVALUATION (128D)")
    print("Fully autoregressive decoding (no teacher forcing)")
    print("="*60)

    results = {}
    for N in [0, 2, 3]:
        label = "Baseline (N=0)" if N == 0 else f"Latency N={N}"
        print(f"Running {label} ...", end="", flush=True)
        dist = evaluate_emb_dist(model, loader, N, emb_weight)
        results[N] = dist
        print(f"  Mean emb dist (128D) = {dist:.4f}")

    print("\nSummary:")
    print(f"  Baseline (N=0) : {results[0]:.4f}")
    print(f"  Latency  N=2   : {results[2]:.4f}  (Δ = {results[2]-results[0]:+.4f})")
    print(f"  Latency  N=3   : {results[3]:.4f}  (Δ = {results[3]-results[0]:+.4f})")
    print(f"\n  (Main-results-table baseline reference: 12.3373)")


if __name__ == "__main__":
    main()
