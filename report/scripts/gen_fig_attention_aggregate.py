"""
Aggregate attention maps over 50 songs and save to report/figures/.

Outputs:
  report/figures/attn_enc_all.png   — encoder self-attention  (6×8, log scale)
  report/figures/attn_cross_all.png — decoder cross-attention (6×8, log scale, raw)
  report/figures/attn_cross_dc.png  — decoder cross-attention (6×8, column-mean removed)
  report/figures/attn_dec_all.png   — decoder self-attention  (6×8, log scale)

Run from repo root:
    python report/scripts/gen_fig_attention_aggregate.py
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from AR.ar_transformer import TransformerModel, MAX_LEN
from AR.attention_visualise import collect_attention_maps
from data.prepare_training_data import break_down_one_song_into_sequences
from utils.constants import DEVICE, INPUT_DIM, NUM_CLASSES_ALL, MEMORY

OUT_DIR    = "report/figures"
N_SONGS    = 50
N_SEQS     = 3
MODEL_PATH = "checkpoints/transformer_model.pth"

os.makedirs(OUT_DIR, exist_ok=True)


def _plot_grid(attn_per_layer, title, cmap, out_path, log=True,
               percentile_clip=None, causal=False):
    """
    6×8 grid: rows = layers, cols = heads.

    percentile_clip : (lo, hi) percentile pair for vmin/vmax, applied to each head's
                      non-masked values.  E.g. (5, 95).
    causal          : if True, only consider the lower-triangle values when computing
                      color range (upper triangle is masked/zero from causal softmax).
    """
    n_layers = len(attn_per_layer)
    n_heads  = attn_per_layer[0].shape[0]
    fig, axes = plt.subplots(n_layers, n_heads,
                             figsize=(3 * n_heads, 3.2 * n_layers),
                             squeeze=False)
    for li in range(n_layers):
        for hi in range(n_heads):
            ax   = axes[li][hi]
            data = attn_per_layer[li][hi].copy()
            if log:
                data = np.log(data + 1e-9)

            # determine colour limits
            if causal:
                T = data.shape[0]
                valid = data[np.tril_indices(T)]           # lower triangle only
            else:
                valid = data.ravel()

            if percentile_clip is not None:
                lo, hi_ = percentile_clip
                vmin = np.percentile(valid, lo)
                vmax = np.percentile(valid, hi_)
            else:
                vmin, vmax = valid.min(), valid.max()

            ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"L{li+1} H{hi+1}", fontsize=8)
            if hi == 0:
                ax.set_ylabel(f"Layer {li+1}", fontsize=8)
            if li == n_layers - 1:
                ax.set_xlabel(f"Head {hi+1}", fontsize=8)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


def _dc_remove(cross_per_layer):
    """
    For each layer/head, subtract the column mean (global-beat bias) from cross-attention,
    shift to non-negative, row-normalise, return list of per-layer arrays.
    """
    result = []
    for li, attn in enumerate(cross_per_layer):           # attn: (H, MAX_LEN, MEMORY)
        H, T, M = attn.shape
        layer_out = np.zeros_like(attn)
        for h in range(H):
            a = attn[h].copy()                            # (T, M)
            a -= a.mean(axis=0, keepdims=True)            # subtract column mean
            a -= a.min(axis=1, keepdims=True)             # shift each row ≥ 0
            row_sum = a.sum(axis=1, keepdims=True)
            row_sum[row_sum == 0] = 1.0
            a = a / row_sum
            layer_out[h] = a
        result.append(layer_out)
    return result


def main():
    print("Loading model…")
    model = TransformerModel(INPUT_DIM, NUM_CLASSES_ALL).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    all_songs = sorted([
        int(f.stem) for f in
        __import__("pathlib").Path("data/pop/melody_chords").glob("*.npz")
    ])
    song_ids = all_songs[:N_SONGS]

    n_enc_layers = len(model.encoder.layers)
    n_dec_layers = len(model.decoder.layers)

    enc_head   = [None] * n_enc_layers
    cross_head = [None] * n_dec_layers
    dec_head   = [None] * n_dec_layers
    count = 0

    for song_id in song_ids:
        npz_path = f"data/pop/melody_chords/{song_id:03d}.npz"
        try:
            melody, target = break_down_one_song_into_sequences(npz_path, test=True)
        except Exception as e:
            print(f"  Skip {song_id}: {e}")
            continue
        if len(melody) == 0:
            continue

        note_counts = [
            sum(1 for beat in mel if any(n > 10 for n in beat[1:]))
            for mel in melody
        ]
        top_idxs = np.argsort(note_counts)[::-1][:N_SEQS]

        for seq_idx in top_idxs:
            mel = melody[seq_idx:seq_idx + 1]
            tgt = target[seq_idx:seq_idx + 1]
            try:
                enc_self, dec_self, dec_cross, _ = collect_attention_maps(model, mel, tgt)
            except Exception as e:
                print(f"  Skip song {song_id} seq {seq_idx}: {e}")
                continue

            for li in range(n_enc_layers):
                e = enc_self[li]
                enc_head[li] = e.copy() if enc_head[li] is None else enc_head[li] + e

            for li in range(n_dec_layers):
                dc = dec_cross[li]
                ds = dec_self[li]
                cross_head[li] = dc.copy() if cross_head[li] is None else cross_head[li] + dc
                dec_head[li]   = ds.copy() if dec_head[li]   is None else dec_head[li]   + ds

            count += 1

        print(f"Song {song_id:03d} done  (sequences so far: {count})")

    enc_head   = [x / count for x in enc_head]
    cross_head = [x / count for x in cross_head]
    dec_head   = [x / count for x in dec_head]

    print(f"\nAggregated {count} sequences from {N_SONGS} songs. Plotting…")

    _plot_grid(enc_head, "Encoder self-attention — all layers × heads (log scale)", "viridis",
               os.path.join(OUT_DIR, "attn_enc_all.png"))

    _plot_grid(cross_head, "Decoder cross-attention — all layers × heads (log scale, raw)", "plasma",
               os.path.join(OUT_DIR, "attn_cross_all.png"))

    _plot_grid(dec_head, "Decoder self-attention — all layers × heads (log scale)", "inferno",
               os.path.join(OUT_DIR, "attn_dec_all.png"),
               percentile_clip=(5, 98), causal=True)

    # DC-removed cross-attention: column mean subtracted, row-normalised, percentile contrast stretch
    cross_dc = _dc_remove(cross_head)
    _plot_grid(cross_dc, "Decoder cross-attention — column mean removed (local alignment)", "plasma",
               os.path.join(OUT_DIR, "attn_cross_dc.png"), log=False,
               percentile_clip=(2, 98))

    print("Done.")


if __name__ == "__main__":
    main()
