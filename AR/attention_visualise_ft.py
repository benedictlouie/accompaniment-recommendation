"""
2D Fourier transform of all attention maps.

Produces three figures (one per attention type):
  enc_ft.png   — encoder self-attention FFT      6 layers × 8 heads
  cross_ft.png — decoder cross-attention FFT     6 layers × 8 heads
  dec_ft.png   — decoder self-attention FFT      6 layers × 8 heads

Each subplot is the log-magnitude 2D FFT of that head's attention map,
averaged over the songs/sequences used for aggregation.

Run:
    python -m AR.attention_visualise_ft [--n_songs N] [--n_seqs N]
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from AR.attention_visualise import collect_attention_maps
from AR.ar_transformer import TransformerModel, MAX_LEN
from data.prepare_training_data import break_down_one_song_into_sequences
from utils.constants import DEVICE, INPUT_DIM, NUM_CLASSES_ALL, MEMORY


# ── FFT helper ────────────────────────────────────────────────────────────────

def _fft2_log_mag(mat: np.ndarray) -> np.ndarray:
    """2D FFT log-magnitude, DC-centred. NaNs treated as zero."""
    mat = np.nan_to_num(mat, nan=0.0)
    mag = np.abs(np.fft.fftshift(np.fft.fft2(mat)))
    return np.log(np.log(mag + 1e-9) - np.log(1e-9) + 1e-9)


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_fourier_grid(attn_per_layer: list[np.ndarray],
                      title: str,
                      cmap: str,
                      out_path: str) -> None:
    """
    6×8 grid of 2D FFT magnitude plots.

    Parameters
    ----------
    attn_per_layer : list of length n_layers, each array (n_heads, Q, K)
    """
    n_layers = len(attn_per_layer)          # 6
    n_heads  = attn_per_layer[0].shape[0]   # 8

    fig, axes = plt.subplots(n_layers, n_heads,
                             figsize=(2.4 * n_heads, 2.6 * n_layers),
                             squeeze=False)

    for li in range(n_layers):
        for hi in range(n_heads):
            ax   = axes[li][hi]
            data = _fft2_log_mag(attn_per_layer[li][hi])   # (Q, K)

            ax.imshow(data, aspect="auto", cmap=cmap,
                      vmin=data.min(), vmax=data.max())
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"L{li+1} H{hi+1}", fontsize=7)

            if hi == 0:
                ax.set_ylabel(f"L{li+1}", fontsize=8)
            if li == n_layers - 1:
                ax.set_xlabel(f"H{hi+1}", fontsize=8)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


def plot_all_fourier(enc_head: list[np.ndarray],
                     cross_head: list[np.ndarray],
                     dec_head: list[np.ndarray],
                     out_dir: str) -> None:
    """Write the three FFT figures to out_dir."""
    os.makedirs(out_dir, exist_ok=True)

    plot_fourier_grid(
        enc_head,
        title="Encoder self-attention — 2D FFT magnitude (6 layers × 8 heads)",
        cmap="viridis",
        out_path=os.path.join(out_dir, "enc_ft.png"),
    )
    plot_fourier_grid(
        cross_head,
        title="Decoder cross-attention — 2D FFT magnitude (6 layers × 8 heads)",
        cmap="plasma",
        out_path=os.path.join(out_dir, "cross_ft.png"),
    )
    plot_fourier_grid(
        dec_head,
        title="Decoder self-attention — 2D FFT magnitude (6 layers × 8 heads)",
        cmap="inferno",
        out_path=os.path.join(out_dir, "dec_ft.png"),
    )


# ── aggregation ───────────────────────────────────────────────────────────────

def aggregate_attention(model: TransformerModel,
                        song_ids: list[int],
                        n_seqs_per_song: int = 3):
    """
    Collect and average attention maps over songs/sequences.

    Returns
    -------
    enc_head   : list[np.ndarray]  (H, MEMORY, MEMORY)  per layer
    cross_head : list[np.ndarray]  (H, MAX_LEN, MEMORY) per layer
    dec_head   : list[np.ndarray]  (H, MAX_LEN, MAX_LEN) per layer
                  upper triangle NaN (causal mask region)
    """
    n_enc = len(model.encoder.layers)
    n_dec = len(model.decoder.layers)

    enc_acc   = [None] * n_enc
    cross_acc = [None] * n_dec
    dec_acc   = [None] * n_dec
    count = 0

    T_dec = None
    upper = None

    for song_id in song_ids:
        npz_path = f"data/pop/melody_chords/{song_id:03d}.npz"
        melody, target = break_down_one_song_into_sequences(npz_path, test=True)
        if len(melody) == 0:
            continue

        note_counts = [
            sum(1 for beat in mel if any(n > 10 for n in beat[1:]))
            for mel in melody
        ]
        top_idxs = np.argsort(note_counts)[::-1][:n_seqs_per_song]

        for seq_idx in top_idxs:
            mel = melody[seq_idx: seq_idx + 1]
            tgt = target[seq_idx: seq_idx + 1]
            try:
                enc_self, dec_self, dec_cross, _ = collect_attention_maps(
                    model, mel, tgt
                )
            except Exception as e:
                print(f"  Skipping song {song_id} seq {seq_idx}: {e}")
                continue

            if upper is None:
                T_dec = dec_self[0].shape[1]
                upper = np.triu(np.ones((T_dec, T_dec), dtype=bool), k=0)

            for li in range(n_enc):
                e = enc_self[li]
                enc_acc[li] = e.copy() if enc_acc[li] is None else enc_acc[li] + e

            for li in range(n_dec):
                dc = dec_cross[li]
                ds = dec_self[li].copy()
                cross_acc[li] = dc.copy() if cross_acc[li] is None else cross_acc[li] + dc
                dec_acc[li]   = ds.copy() if dec_acc[li]   is None else dec_acc[li]   + ds

            count += 1

        print(f"Song {song_id:03d} done  (sequences so far: {count})")

    if count == 0:
        raise RuntimeError("No sequences were successfully processed.")

    enc_head   = [x / count for x in enc_acc]
    cross_head = [x / count for x in cross_acc]
    dec_head   = [x / count for x in dec_acc]

    # Mask causal region with NaN so FFT treats it as zero
    for li in range(n_dec):
        dec_head[li][:, upper] = np.nan

    return enc_head, cross_head, dec_head


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="2D Fourier transform of all attention maps"
    )
    parser.add_argument("--n_songs", type=int, default=50,
                        help="Number of songs to aggregate over")
    parser.add_argument("--n_seqs",  type=int, default=3,
                        help="Sequences per song")
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
    plot_all_fourier(enc_head, cross_head, dec_head, out_dir=out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
