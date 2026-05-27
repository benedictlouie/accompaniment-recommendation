"""
Attention map visualiser for the AR Transformer.

Produces three figures per example:
  1. Decoder cross-attention  — which melody beats each output chord attends to
  2. Decoder self-attention   — which previous chord tokens each chord attends to
  3. Encoder self-attention   — how melody beats relate to one another

Run:
    python -m AR.attention_visualise [--song SONG_NUM] [--layer LAYER]

Arguments
---------
--song   : integer song id in data/pop/melody_chords/ (default 707)
--layer  : which decoder layer (0-indexed) to plot in detail (default: all)
--heads  : show per-head plots instead of averaging (flag)
"""

import argparse
import sys
import os

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")          # works headless; swap to "TkAgg" / "MacOSX" for interactive
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from AR.ar_transformer import TransformerModel, MAX_LEN
from data.prepare_training_data import break_down_one_song_into_sequences
from utils.constants import (
    DEVICE, INPUT_DIM, NUM_CLASSES_ALL, CHORD_CLASSES_ALL, MEMORY, ROOTS
)

# ── attention capture wrapper ─────────────────────────────────────────────────

class _MHACapture(nn.Module):
    """Wraps a nn.MultiheadAttention and stores the last per-head weight tensor."""

    def __init__(self, mha: nn.MultiheadAttention):
        super().__init__()
        self.mha = mha
        self.weights = None   # (B, H, T_q, T_k)

    def __getattr__(self, name):
        # Forward any attribute access the Transformer layers need (e.g. batch_first)
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.mha, name)

    def forward(self, query, key, value, **kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False
        out, w = self.mha(query, key, value, **kwargs)
        self.weights = w.detach().cpu()            # save for inspection
        return out, w


def _install_capture_wrappers(model: TransformerModel):
    """Replace all MHA modules with _MHACapture and return handles for removal."""
    captures = {"enc_self": [], "dec_self": [], "dec_cross": []}

    for layer in model.encoder.layers:
        cap = _MHACapture(layer.self_attn)
        layer.self_attn = cap
        captures["enc_self"].append(cap)

    for layer in model.decoder.layers:
        cap_self = _MHACapture(layer.self_attn)
        cap_cross = _MHACapture(layer.multihead_attn)
        layer.self_attn = cap_self
        layer.multihead_attn = cap_cross
        captures["dec_self"].append(cap_self)
        captures["dec_cross"].append(cap_cross)

    return captures


def _remove_capture_wrappers(model: TransformerModel, captures: dict):
    """Restore original MHA modules."""
    for layer, cap in zip(model.encoder.layers, captures["enc_self"]):
        layer.self_attn = cap.mha
    for layer, cap_self, cap_cross in zip(
        model.decoder.layers, captures["dec_self"], captures["dec_cross"]
    ):
        layer.self_attn = cap_self.mha
        layer.multihead_attn = cap_cross.mha


# ── forward pass with attention collection ───────────────────────────────────

def collect_attention_maps(model: TransformerModel, melody: np.ndarray, target: np.ndarray):
    """
    Run a teacher-forced forward pass and collect attention maps.

    Returns
    -------
    enc_self  : list[np.ndarray]   shape (H, MEMORY, MEMORY) per layer
    dec_self  : list[np.ndarray]   shape (H, MAX_LEN, MAX_LEN) per layer
    dec_cross : list[np.ndarray]   shape (H, MAX_LEN, MEMORY) per layer
    pred_chords : list[str]        predicted chord name per output position
    """
    captures = _install_capture_wrappers(model)

    mel_t = torch.tensor(melody, dtype=torch.float32).to(DEVICE)   # (1, T, F)
    tgt_t = torch.tensor(target, dtype=torch.long).to(DEVICE)       # (1, MAX_LEN, 1)

    model.eval()
    # Build causal mask matching training (diagonal=1: position t can attend to itself)
    causal_mask = nn.Transformer.generate_square_subsequent_mask(MAX_LEN, device=DEVICE)

    # Patch model.decoder.forward to inject the mask without modifying model code
    original_decoder_forward = model.decoder.forward
    def _masked_decoder_forward(tgt, memory, **kwargs):
        kwargs["tgt_mask"] = causal_mask
        return original_decoder_forward(tgt, memory, **kwargs)
    model.decoder.forward = _masked_decoder_forward

    prev_fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        with torch.no_grad():
            logits = model(mel_t, tgt_t, use_teacher=True)          # (1, MAX_LEN, V)
            preds = logits.argmax(dim=-1)[0].cpu().numpy()          # (MAX_LEN,)
    finally:
        torch.backends.mha.set_fastpath_enabled(prev_fastpath)
        model.decoder.forward = original_decoder_forward

    # Collect weights (take batch=0, convert to numpy)
    enc_self  = [cap.weights[0].numpy() for cap in captures["enc_self"]]
    dec_self  = [cap.weights[0].numpy() for cap in captures["dec_self"]]
    dec_cross = [cap.weights[0].numpy() for cap in captures["dec_cross"]]

    _remove_capture_wrappers(model, captures)

    pred_chords = [str(CHORD_CLASSES_ALL[p]) for p in preds]
    return enc_self, dec_self, dec_cross, pred_chords


# ── plotting helpers ──────────────────────────────────────────────────────────

def _heatmap(ax, data, xticklabels, yticklabels, title, xlabel, ylabel,
             fontsize=7, cmap="Blues"):
    """Draw a single attention heatmap on ax."""
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=np.nanmin(data), vmax=np.nanmax(data))
    ax.set_xticks(range(len(xticklabels)))
    ax.set_yticks(range(len(yticklabels)))
    ax.set_xticklabels(xticklabels, rotation=90, fontsize=fontsize)
    ax.set_yticklabels(yticklabels, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9)
    return im


def _beat_labels(melody: np.ndarray) -> list[str]:
    """
    melody: (MEMORY, INPUT_DIM) — col 0 is strong-beat flag, cols 1-4 are MIDI notes.
    Returns one label per beat: the unique pitch names played (e.g. "E/G#"), or "-" for rest.
    """
    labels = []
    for beat in melody:
        notes = beat[1:]                              # cols 1-4: MIDI numbers
        pitches = sorted({int(n) % 12 for n in notes if n > 10})
        labels.append("/".join(ROOTS[p] for p in pitches) if pitches else "-")
    return labels


def _chord_labels(names: list[str]) -> list[str]:
    return list(names)


# ── main plot functions ───────────────────────────────────────────────────────

def plot_cross_attention(dec_cross, pred_chords, melody: np.ndarray,
                         per_head=False, out_path="attention_cross.png"):
    """Plot decoder cross-attention for all 6 layers (averaged or per-head)."""
    n_layers = len(dec_cross)
    beat_lbl  = _beat_labels(melody)
    chord_lbl = _chord_labels(pred_chords)

    if per_head:
        n_heads = dec_cross[0].shape[0]
        fig, axes = plt.subplots(n_layers, n_heads,
                                 figsize=(3 * n_heads, 3 * n_layers),
                                 squeeze=False)
        for li in range(n_layers):
            for hi in range(n_heads):
                _heatmap(axes[li][hi], dec_cross[li][hi],
                         xticklabels=beat_lbl,
                         yticklabels=chord_lbl,
                         title=f"L{li} H{hi}",
                         xlabel="Melody beat",
                         ylabel="Output chord step")
    else:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10), squeeze=False)
        for li in range(n_layers):
            ax = axes[li // 3][li % 3]
            avg = dec_cross[li].mean(axis=0)   # (MAX_LEN, MEMORY)
            _heatmap(ax, avg,
                     xticklabels=beat_lbl,
                     yticklabels=chord_lbl,
                     title=f"Decoder cross-attention — layer {li}",
                     xlabel="Melody beat (encoder position)",
                     ylabel="Output chord step")

    fig.suptitle("Decoder Cross-Attention\n(rows = output chord, cols = melody input beat)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


def plot_self_attention(dec_self, pred_chords,
                        per_head=False, out_path="attention_self.png"):
    """Plot decoder self-attention for all 6 layers."""
    n_layers = len(dec_self)
    chord_lbl = _chord_labels(pred_chords)

    if per_head:
        n_heads = dec_self[0].shape[0]
        fig, axes = plt.subplots(n_layers, n_heads,
                                 figsize=(2.5 * n_heads, 2.5 * n_layers),
                                 squeeze=False)
        for li in range(n_layers):
            for hi in range(n_heads):
                _heatmap(axes[li][hi], dec_self[li][hi],
                         xticklabels=chord_lbl,
                         yticklabels=chord_lbl,
                         title=f"L{li} H{hi}",
                         xlabel="Key (previous chord)",
                         ylabel="Query (current chord)")
    else:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10), squeeze=False)
        for li in range(n_layers):
            ax = axes[li // 3][li % 3]
            avg = dec_self[li].mean(axis=0)    # (MAX_LEN, MAX_LEN)
            _heatmap(ax, avg,
                     xticklabels=chord_lbl,
                     yticklabels=chord_lbl,
                     title=f"Decoder self-attention — layer {li}",
                     xlabel="Key (previous chord step)",
                     ylabel="Query (current chord step)",
                     cmap="Purples")

    fig.suptitle("Decoder Self-Attention\n(rows = query chord, cols = attended key chord)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


def plot_encoder_attention(enc_self, melody: np.ndarray,
                           per_head=False, out_path="attention_encoder.png"):
    """Plot encoder self-attention — how melody beats relate to each other."""
    n_layers = len(enc_self)
    beat_lbl = _beat_labels(melody)

    if per_head:
        n_heads = enc_self[0].shape[0]
        fig, axes = plt.subplots(n_layers, n_heads,
                                 figsize=(2.5 * n_heads, 2.5 * n_layers),
                                 squeeze=False)
        for li in range(n_layers):
            for hi in range(n_heads):
                _heatmap(axes[li][hi], enc_self[li][hi],
                         xticklabels=beat_lbl,
                         yticklabels=beat_lbl,
                         title=f"L{li} H{hi}",
                         xlabel="Key beat",
                         ylabel="Query beat")
    else:
        fig, axes = plt.subplots(2, 3, figsize=(16, 10), squeeze=False)
        for li in range(n_layers):
            ax = axes[li // 3][li % 3]
            avg = enc_self[li].mean(axis=0)    # (MEMORY, MEMORY)
            _heatmap(ax, avg,
                     xticklabels=beat_lbl,
                     yticklabels=beat_lbl,
                     title=f"Encoder self-attention — layer {li}",
                     xlabel="Key (melody beat)",
                     ylabel="Query (melody beat)",
                     cmap="Greens")

    fig.suptitle("Encoder Self-Attention\n(how melody beats attend to each other)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


def _remove_dominant(enc, cross, dec_self_raw, lower):
    """
    Row 1 — remove diagonal (self-attn) / column mean (cross-attn), renormalise.
    Returns log-scale versions ready for plotting.
    """
    # Encoder: zero diagonal, renormalise rows
    enc_r = enc.copy()
    np.fill_diagonal(enc_r, 0.0)
    row_sum = enc_r.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    enc_r = np.log(enc_r / row_sum + 1e-9)

    # Cross-attn: subtract column mean (removes globally dominant beats), clip, renormalise
    cross_r = cross.copy()
    cross_r -= cross_r.mean(axis=0, keepdims=True)
    cross_r -= cross_r.min(axis=1, keepdims=True)           # shift each row ≥ 0
    row_sum = cross_r.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    cross_r = np.log(cross_r / row_sum + 1e-9)

    # Decoder self-attn: zero diagonal within valid region, renormalise rows
    ds_r = dec_self_raw.copy()
    ds_r[lower] = 0.0
    np.fill_diagonal(ds_r, 0.0)
    row_sum = np.nansum(ds_r, axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    ds_r = np.where(~lower, np.log(ds_r / row_sum + 1e-9), np.nan)

    return enc_r, cross_r, ds_r


def plot_single_layer_detail(enc_self, dec_self, dec_cross, pred_chords,
                              melody: np.ndarray, layer=None,
                              out_path="attention_detail.png"):
    """
    2×3 grid: top row = raw log attention, bottom row = dominant pattern removed + renormalised.
    """
    beat_lbl  = _beat_labels(melody)
    chord_lbl = _chord_labels(pred_chords)

    enc_raw   = np.stack(enc_self).mean(axis=(0, 1))
    cross_raw = np.stack(dec_cross).mean(axis=(0, 1))
    ds_raw    = np.stack(dec_self).mean(axis=(0, 1))
    T = ds_raw.shape[0]
    lower = np.tril(np.ones((T, T), dtype=bool), k=-1)

    # Top row: log scale
    enc_avg       = np.log(enc_raw  + 1e-9)
    dec_cross_avg = np.log(cross_raw + 1e-9)
    dec_self_avg  = np.log(ds_raw + 1e-9)

    # Bottom row: dominant removed + renormalised
    enc_res, cross_res, ds_res = _remove_dominant(enc_raw, cross_raw, ds_raw, lower)

    n_layers = len(enc_self)
    n_heads  = enc_self[0].shape[0]

    fig, axes = plt.subplots(2, 3, figsize=(22, 14))

    # ── top row ───────────────────────────────────────────────────────────────
    _heatmap(axes[0][0], enc_avg,       beat_lbl,  beat_lbl,  "Encoder self-attn",         "Key beat",       "Query beat",        cmap="viridis")
    _heatmap(axes[0][1], dec_cross_avg, beat_lbl,  chord_lbl, "Decoder cross-attn",         "Melody beat",    "Output chord step", cmap="plasma")
    _heatmap(axes[0][2], dec_self_avg,  chord_lbl, chord_lbl, "Decoder self-attn",          "Key chord step", "Query chord step",  cmap="inferno")

    # ── bottom row ────────────────────────────────────────────────────────────
    _heatmap(axes[1][0], enc_res,   beat_lbl,  beat_lbl,  "Encoder (diagonal removed)",         "Key beat",       "Query beat",        cmap="viridis")
    _heatmap(axes[1][1], cross_res, beat_lbl,  chord_lbl, "Cross-attn (global stripes removed)", "Melody beat",    "Output chord step", cmap="plasma")
    _heatmap(axes[1][2], ds_res,    chord_lbl, chord_lbl, "Decoder self-attn (diag removed)",   "Key chord step", "Query chord step",  cmap="inferno")

    fig.suptitle(f"Attention maps (log scale) — averaged over all {n_layers} layers and {n_heads} heads",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


# ── focused plot functions ───────────────────────────────────────────────────

def plot_all_heads_layers(attn_per_layer, pred_chords, melody,
                          beat_lbl=None, chord_lbl=None,
                          title="Attention — all layers × heads",
                          cmap="plasma",
                          out_path="attn_all.png"):
    """6×8 grid: rows = layers, cols = heads."""
    n_layers = len(attn_per_layer)
    n_heads  = attn_per_layer[0].shape[0]
    if beat_lbl  is None: beat_lbl  = _beat_labels(melody)
    if chord_lbl is None: chord_lbl = _chord_labels(pred_chords)

    fig, axes = plt.subplots(n_layers, n_heads,
                             figsize=(3 * n_heads, 3.2 * n_layers),
                             squeeze=False)

    for li in range(n_layers):
        for hi in range(n_heads):
            ax   = axes[li][hi]
            data = np.log(attn_per_layer[li][hi] + 1e-9)
            ax.imshow(data, aspect="auto", cmap=cmap,
                      vmin=np.nanmin(data), vmax=np.nanmax(data))
            ax.set_xticks([]); ax.set_yticks([])
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


def plot_cross_heads(dec_cross, pred_chords, melody, layer=-1,
                     out_path="cross_heads.png"):
    """2×4 grid — one subplot per head for a single decoder layer."""
    attn    = dec_cross[layer]          # (H, MAX_LEN, MEMORY)
    n_heads = attn.shape[0]
    beat_lbl  = _beat_labels(melody) if melody is not None \
                else [str(i+1) for i in range(attn.shape[2])]
    chord_lbl = list(pred_chords) if pred_chords is not None \
                else [str(i+1) for i in range(attn.shape[1])]
    layer_idx = layer % len(dec_cross)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10), squeeze=False)
    for h in range(n_heads):
        ax   = axes[h // 4][h % 4]
        data = np.log(attn[h] + 1e-9)
        _heatmap(ax, data, beat_lbl, chord_lbl,
                 f"Head {h+1}", "Melody beat", "Output chord step",
                 cmap="plasma", fontsize=6)

    fig.suptitle(f"Decoder cross-attention — per head  (layer {layer_idx+1})", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


def plot_cross_layers(dec_cross, pred_chords, melody,
                      beat_lbl=None, chord_lbl=None,
                      out_path="cross_layers.png"):
    """2×3 grid — cross-attention averaged over heads, one subplot per layer."""
    n_layers = len(dec_cross)
    if beat_lbl  is None: beat_lbl  = _beat_labels(melody)
    if chord_lbl is None: chord_lbl = _chord_labels(pred_chords)

    fig, axes = plt.subplots(2, 3, figsize=(20, 12), squeeze=False)
    for li in range(n_layers):
        ax  = axes[li // 3][li % 3]
        avg = dec_cross[li].mean(axis=0)          # (MAX_LEN, MEMORY)
        _heatmap(ax, np.log(avg + 1e-9), beat_lbl, chord_lbl,
                 f"Layer {li+1}", "Melody beat", "Output chord step",
                 cmap="plasma", fontsize=6)

    fig.suptitle("Cross-attention through layers  (heads averaged, log scale)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


def plot_entropy(dec_cross, dec_self, out_path="attention_entropy.png"):
    """
    Line plot: attention entropy per output chord step.
    Low entropy = model focuses on a few positions; high = spread out.
    """
    cross_avg = np.stack(dec_cross).mean(axis=(0, 1))   # (MAX_LEN, MEMORY)
    ds_avg    = np.stack(dec_self).mean(axis=(0, 1))    # (MAX_LEN, MAX_LEN)
    T = ds_avg.shape[0]
    lower = np.tril(np.ones((T, T), dtype=bool), k=-1)

    # Cross-attention entropy per chord step
    cross_ent = -np.sum(cross_avg * np.log(cross_avg + 1e-9), axis=1)  # (MAX_LEN,)

    # Decoder self-attn entropy: only valid (unmasked) positions
    ds_ent = []
    for t in range(T):
        row = ds_avg[t].copy()
        row[lower[t]] = 0.0
        row[t]        = 0.0                # exclude diagonal
        total = row.sum()
        if total > 1e-12:
            row = row / total
        ds_ent.append(-np.sum(row * np.log(row + 1e-9)))

    x = np.arange(1, T + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(x, cross_ent, marker='o', markersize=4, lw=1.5, color='tab:orange')
    axes[0].set_xlabel("Output chord step", fontsize=11)
    axes[0].set_ylabel("Entropy (nats)", fontsize=11)
    axes[0].set_title("Cross-attention entropy\n(low = attends to few melody beats)", fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, ds_ent, marker='o', markersize=4, lw=1.5, color='tab:purple')
    axes[1].set_xlabel("Output chord step", fontsize=11)
    axes[1].set_ylabel("Entropy (nats)", fontsize=11)
    axes[1].set_title("Decoder self-attention entropy\n(low = attends to few prior chords)", fontsize=10)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Attention entropy — averaged over all layers and heads", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


# ── aggregate over many songs/sequences ──────────────────────────────────────

def plot_aggregate(model, song_ids, n_seqs_per_song=3, out_dir="AR"):
    n_enc_layers = len(model.encoder.layers)
    n_dec_layers = len(model.decoder.layers)

    enc_head  = [None] * n_enc_layers   # (H, MEMORY, MEMORY)
    cross_head = [None] * n_dec_layers  # (H, MAX_LEN, MEMORY)
    dec_head  = [None] * n_dec_layers   # (H, MAX_LEN, MAX_LEN)
    count = 0

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
            mel = melody[seq_idx:seq_idx+1]
            tgt = target[seq_idx:seq_idx+1]
            try:
                enc_self, dec_self, dec_cross, _ = collect_attention_maps(model, mel, tgt)
            except Exception as e:
                print(f"  Skipping song {song_id} seq {seq_idx}: {e}")
                continue

            for li in range(n_enc_layers):
                e = enc_self[li]   # (H, MEMORY, MEMORY)
                enc_head[li] = e.copy() if enc_head[li] is None else enc_head[li] + e

            for li in range(n_dec_layers):
                dc = dec_cross[li]              # (H, MAX_LEN, MEMORY)
                ds = dec_self[li].copy()        # (H, MAX_LEN, MAX_LEN)
                # upper triangle is already 0 from causal softmax — just accumulate as-is
                cross_head[li] = dc.copy() if cross_head[li] is None else cross_head[li] + dc
                dec_head[li]   = ds.copy() if dec_head[li]   is None else dec_head[li]   + ds

            count += 1

        print(f"Song {song_id:03d} done  (total sequences so far: {count})")

    enc_head   = [x / count for x in enc_head]
    cross_head = [x / count for x in cross_head]
    dec_head   = [x / count for x in dec_head]

    n_s = len(song_ids)
    beat_lbl  = [str(i+1) for i in range(MEMORY)]
    chord_lbl = [str(i+1) for i in range(MAX_LEN)]

    plot_all_heads_layers(enc_head,   pred_chords=None, melody=None,
                          beat_lbl=beat_lbl,  chord_lbl=beat_lbl,
                          title="Encoder self-attention — all layers × heads",
                          cmap="viridis",
                          out_path=os.path.join(out_dir, f"agg_{n_s}s_enc_all.png"))

    plot_all_heads_layers(cross_head, pred_chords=None, melody=None,
                          beat_lbl=beat_lbl,  chord_lbl=chord_lbl,
                          title="Decoder cross-attention — all layers × heads",
                          cmap="plasma",
                          out_path=os.path.join(out_dir, f"agg_{n_s}s_cross_all.png"))

    plot_all_heads_layers(dec_head,   pred_chords=None, melody=None,
                          beat_lbl=chord_lbl, chord_lbl=chord_lbl,
                          title="Decoder self-attention — all layers × heads",
                          cmap="inferno",
                          out_path=os.path.join(out_dir, f"agg_{n_s}s_dec_all.png"))

    print(f"Saved to {out_dir}/")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualise transformer attention maps")
    parser.add_argument("--n_songs", type=int, default=50, help="How many songs to aggregate over")
    parser.add_argument("--n_seqs",  type=int, default=3,  help="Sequences per song")
    args = parser.parse_args()

    # ── load model ────────────────────────────────────────────────────────────
    print("Loading model…")
    model = TransformerModel(INPUT_DIM, NUM_CLASSES_ALL).to(DEVICE)
    ckpt = torch.load("checkpoints/transformer_model.pth", map_location=DEVICE)
    model.load_state_dict(ckpt)

    ar_dir   = os.path.dirname(os.path.abspath(__file__))
    all_songs = sorted([
        int(f.stem) for f in
        __import__("pathlib").Path("data/pop/melody_chords").glob("*.npz")
    ])
    song_ids = all_songs[:args.n_songs]
    plot_aggregate(model, song_ids, n_seqs_per_song=args.n_seqs, out_dir=ar_dir)
    print("Done.")

    print("Done.")


if __name__ == "__main__":
    main()
