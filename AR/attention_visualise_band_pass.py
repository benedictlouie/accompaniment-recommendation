"""
Filtered attention maps reconstructed in the spatial domain.

Three filter modes:
  highpass    — remove central disc (kills global/smooth structure)
  lowpass     — keep only central disc (smooth global component)
  crossnotch  — remove only the central row + column exactly
                (kills global beat/query dominance, keeps all 2D structure)

Run:
    python -m AR.attention_visualise_band_pass [--n_songs N] [--n_seqs N]
                                               [--mode highpass|lowpass|crossnotch]
                                               [--radius R]
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from AR.attention_visualise_ft import aggregate_attention
from AR.ar_transformer import TransformerModel, MAX_LEN
from utils.constants import DEVICE, INPUT_DIM, NUM_CLASSES_ALL, MEMORY
import torch


# ── filters ───────────────────────────────────────────────────────────────────

def _apply_filter(mat: np.ndarray, mode: str, radius: float) -> np.ndarray:
    mat = np.nan_to_num(mat, nan=0.0)
    F   = np.fft.fftshift(np.fft.fft2(mat))
    H, W = F.shape
    cy, cx = H // 2, W // 2

    if mode in ("highpass", "lowpass"):
        r_px = radius * min(H, W) / 2
        ys, xs = np.ogrid[:H, :W]
        disc = (ys - cy) ** 2 + (xs - cx) ** 2 <= r_px ** 2
        if mode == "highpass":
            F[disc] = 0.0
        else:
            F[~disc] = 0.0

    elif mode == "crossnotch":
        F[cy, :] = 0.0   # central row  (zero col-frequency = globally popular cols)
        F[:, cx] = 0.0   # central col  (zero row-frequency = globally popular rows)

    return np.real(np.fft.ifft2(np.fft.ifftshift(F)))


def _doublecentre(mat: np.ndarray) -> np.ndarray:
    """Subtract row means and column means, add grand mean back (double-centering)."""
    mat = np.nan_to_num(mat, nan=0.0)
    return mat - mat.mean(axis=1, keepdims=True) - mat.mean(axis=0, keepdims=True) + mat.mean()


# ── plotting ──────────────────────────────────────────────────────────────────

CMAPS = {
    "enc":   "viridis",
    "cross": "plasma",
    "dec":   "inferno",
}


def plot_filtered_grid(attn_per_layer: list[np.ndarray],
                       title: str,
                       out_path: str,
                       mode: str,
                       radius: float,
                       cmap: str = "plasma") -> None:
    """6×8 grid of filtered+reconstructed attention maps."""
    n_layers = len(attn_per_layer)
    n_heads  = attn_per_layer[0].shape[0]

    fig, axes = plt.subplots(n_layers, n_heads,
                             figsize=(2.4 * n_heads, 2.6 * n_layers),
                             squeeze=False)

    for li in range(n_layers):
        for hi in range(n_heads):
            ax   = axes[li][hi]
            if mode == "doublecentre":
                data = _doublecentre(attn_per_layer[li][hi])
            else:
                data = _apply_filter(attn_per_layer[li][hi], mode, radius)

            vmin, vmax = np.percentile(data, 2), np.percentile(data, 98)

            ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"L{li+1} H{hi+1}", fontsize=7)
            if hi == 0:
                ax.set_ylabel(f"L{li+1}", fontsize=8)
            if li == n_layers - 1:
                ax.set_xlabel(f"H{hi+1}", fontsize=8)

    fig.suptitle(f"{title}  [{mode}]", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


def plot_all(enc_head, cross_head, dec_head, out_dir, mode, radius):
    os.makedirs(out_dir, exist_ok=True)
    suffix = {"highpass": "hp", "lowpass": "lo", "crossnotch": "cr", "doublecentre": "dc"}[mode]

    plot_filtered_grid(enc_head,   "Encoder self-attention",  os.path.join(out_dir, f"enc_{suffix}.png"),   mode, radius, cmap=CMAPS["enc"])
    plot_filtered_grid(cross_head, "Decoder cross-attention", os.path.join(out_dir, f"cross_{suffix}.png"), mode, radius, cmap=CMAPS["cross"])
    plot_filtered_grid(dec_head,   "Decoder self-attention",  os.path.join(out_dir, f"dec_{suffix}.png"),   mode, radius, cmap=CMAPS["dec"])


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_songs", type=int,   default=50)
    parser.add_argument("--n_seqs",  type=int,   default=3)
    parser.add_argument("--mode",    type=str,   default="doublecentre",
                        choices=["highpass", "lowpass", "crossnotch", "doublecentre"])
    parser.add_argument("--radius",  type=float, default=0.2,
                        help="Disc radius for highpass/lowpass (fraction of min(H,W)/2)")
    args = parser.parse_args()

    print("Loading model…")
    model = TransformerModel(INPUT_DIM, NUM_CLASSES_ALL).to(DEVICE)
    ckpt  = torch.load("checkpoints/transformer_model.pth", map_location=DEVICE)
    model.load_state_dict(ckpt)

    all_songs = sorted([
        int(f.stem) for f in
        __import__("pathlib").Path("data/pop/melody_chords").glob("*.npz")
    ])
    song_ids = all_songs[: args.n_songs]

    print(f"Aggregating over {len(song_ids)} songs ({args.n_seqs} seqs each)…")
    enc_head, cross_head, dec_head = aggregate_attention(
        model, song_ids, n_seqs_per_song=args.n_seqs
    )

    out_dir = os.path.dirname(os.path.abspath(__file__))
    plot_all(enc_head, cross_head, dec_head, out_dir, args.mode, args.radius)
    print("Done.")


if __name__ == "__main__":
    main()
