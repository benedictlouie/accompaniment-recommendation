"""
Prediction evolution visualiser for the AR Transformer.

At each autoregressive decoding step t, the decoder computes representations
for ALL positions 0..t. This script captures those full outputs and shows how
the model's belief about bar x changes as more bars are generated.

Produces two figures:
  1. Heatmap  — P(correct chord | context length t) for every bar x, every step t ≥ x
  2. Top-k    — at each decoding step t, the top-3 predicted chords for bar x

Run:
    python -m AR.prediction_evolution [--song SONG_NUM] [--seq SEQ_IDX] [--bar BAR]
"""

import argparse
import sys
import os

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from AR.ar_transformer import TransformerModel, MAX_LEN
from data.prepare_training_data import break_down_one_song_into_sequences
from utils.constants import DEVICE, INPUT_DIM, NUM_CLASSES_ALL, CHORD_CLASSES_ALL, MEMORY, ROOTS


# ── core: collect per-step full-output logits ─────────────────────────────────

def collect_evolution(model: TransformerModel, melody: np.ndarray, target: np.ndarray):
    """
    Run AR decoding and, at every step t, capture logits for ALL positions 0..t.

    Returns
    -------
    logits_evolution : np.ndarray  shape (MAX_LEN, MAX_LEN, V)
        logits_evolution[t, x, :] = logit vector for bar x at decoding step t.
        Positions x > t are filled with NaN (not yet reached).
    ar_preds : list[int]   greedy predicted class for each bar (normal AR output)
    target_seq : list[int] ground-truth class for each bar
    """
    model.eval()
    mel_t = torch.tensor(melody, dtype=torch.float32).to(DEVICE)   # (1, MEMORY, F)
    V = NUM_CLASSES_ALL

    logits_evolution = np.full((MAX_LEN, MAX_LEN, V), np.nan, dtype=np.float32)
    ar_preds = []

    with torch.no_grad():
        # Encode melody once
        enc_in = mel_t
        memory_enc = model.encoder(
            model.feature_to_embedding(
                torch.where(enc_in > 10, enc_in % 12, enc_in)
            ) + model.pos_encoder
        )   # (1, MEMORY, d_model)

        # AR loop — same as model.forward but we also read non-last positions
        tgt_emb = torch.zeros(1, 1, model.d_model, device=DEVICE)

        for t in range(MAX_LEN):
            out = model.decoder(tgt=tgt_emb, memory=memory_enc)  # (1, t+1, d_model)
            step_logits = model.fc_out(out[0])                   # (t+1, V)

            # Store logits for all positions seen so far
            logits_evolution[t, : t + 1, :] = step_logits.cpu().numpy()

            # Greedy prediction for bar t (last position)
            pred_t = step_logits[-1].argmax().item()
            ar_preds.append(pred_t)

            # Next input embedding: weighted sum through vocab (soft embedding)
            probs_t = torch.softmax(step_logits[-1], dim=-1)     # (V,)
            new_emb = (probs_t @ model.embedding_output.weight).unsqueeze(0).unsqueeze(0)  # (1,1,d)
            tgt_emb = torch.cat([tgt_emb, new_emb], dim=1)

    target_seq = list(target.squeeze(-1).flatten()[:MAX_LEN])
    return logits_evolution, ar_preds, [int(x) for x in target_seq]


# ── plot 1: P(correct) heatmap ────────────────────────────────────────────────

def plot_correct_prob_heatmap(logits_evolution, target_seq, ar_preds,
                               out_path="prediction_evolution_heatmap.png"):
    """
    Rows   = bars (x-axis label: bar index)
    Columns = decoding step t
    Colour  = P(correct chord for bar x) given context length t
    Only the lower-triangular part (t ≥ x) is valid.
    """
    MAX_LEN, _, V = logits_evolution.shape
    prob_matrix = np.full((MAX_LEN, MAX_LEN), np.nan)

    for x in range(MAX_LEN):
        true_cls = target_seq[x]
        for t in range(x, MAX_LEN):
            logit_row = logits_evolution[t, x, :]
            if np.isnan(logit_row).all():
                continue
            probs = _softmax(logit_row)
            prob_matrix[x, t] = probs[true_cls]

    fig, ax = plt.subplots(figsize=(14, 10))
    masked = np.ma.masked_invalid(prob_matrix)
    im = ax.imshow(masked, aspect="auto", origin="upper",
                   cmap="RdYlGn", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="P(correct chord)")

    # Bar labels
    bar_labels = [str(CHORD_CLASSES_ALL[target_seq[x]]) for x in range(MAX_LEN)]
    pred_labels = [str(CHORD_CLASSES_ALL[ar_preds[x]]) for x in range(MAX_LEN)]
    ytick_labels = [f"{x}: {bar_labels[x]}" for x in range(MAX_LEN)]

    ax.set_yticks(range(MAX_LEN))
    ax.set_yticklabels(ytick_labels, fontsize=6)
    ax.set_xticks(range(MAX_LEN))
    ax.set_xticklabels([str(t) for t in range(MAX_LEN)], rotation=90, fontsize=6)
    ax.set_xlabel("Decoding step t (context length)", fontsize=10)
    ax.set_ylabel("Bar x (ground truth chord)", fontsize=10)
    ax.set_title(
        "P(correct chord for bar x) at each decoding step t ≥ x\n"
        "Diagonal = when bar x is first predicted; rightward = more context",
        fontsize=11
    )

    # Diagonal line marks first prediction
    diag_x = np.arange(MAX_LEN)
    ax.plot(diag_x, diag_x, color="white", lw=1.5, ls="--", alpha=0.7, label="First prediction")
    ax.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


# ── plot 2: top-k evolution for a single bar ──────────────────────────────────

def plot_bar_topk_evolution(logits_evolution, target_seq, ar_preds, bar_x,
                             k=5, out_path="prediction_evolution_bar.png"):
    """
    For bar x, show how the top-k probabilities evolve across decoding steps t ≥ x.
    Also draws a reference line for P(correct chord).
    """
    MAX_LEN = logits_evolution.shape[0]
    true_cls = target_seq[bar_x]
    steps = list(range(bar_x, MAX_LEN))

    # Collect top-k classes (union across all steps for bar x)
    all_topk_classes = set()
    probs_by_step = {}
    for t in steps:
        row = logits_evolution[t, bar_x, :]
        if np.isnan(row).all():
            continue
        probs = _softmax(row)
        probs_by_step[t] = probs
        topk_idx = np.argsort(probs)[::-1][:k]
        all_topk_classes.update(topk_idx.tolist())

    all_topk_classes.discard(true_cls)   # drawn separately
    topk_list = sorted(all_topk_classes,
                       key=lambda c: -probs_by_step[bar_x][c]
                       if bar_x in probs_by_step else 0)[:k]

    fig, ax = plt.subplots(figsize=(max(10, len(steps) * 0.4 + 2), 5))

    # True chord probability
    true_probs = [probs_by_step[t][true_cls] if t in probs_by_step else np.nan
                  for t in steps]
    ax.plot(steps, true_probs, color="green", lw=2.5, marker="o", markersize=4,
            label=f"✓ {CHORD_CLASSES_ALL[true_cls]} (correct)", zorder=5)

    # First-predicted chord (diagonal)
    if bar_x in probs_by_step:
        first_pred = int(np.argmax(logits_evolution[bar_x, bar_x, :]))
        if first_pred != true_cls:
            fp_probs = [probs_by_step[t][first_pred] if t in probs_by_step else np.nan
                        for t in steps]
            ax.plot(steps, fp_probs, color="red", lw=1.8, marker="s", markersize=3,
                    ls="--", label=f"First pred: {CHORD_CLASSES_ALL[first_pred]}", zorder=4)

    # Other top-k competitors
    cmap = plt.cm.tab10
    for ci, cls in enumerate(topk_list[:k-1]):
        if cls == true_cls:
            continue
        plist = [probs_by_step[t][cls] if t in probs_by_step else np.nan for t in steps]
        ax.plot(steps, plist, color=cmap(ci + 2), lw=1.2, marker=".", markersize=3,
                alpha=0.7, label=str(CHORD_CLASSES_ALL[cls]))

    # Mark step bar_x (when bar is first predicted)
    ax.axvline(bar_x, color="black", lw=1.2, ls=":", alpha=0.6, label=f"Step {bar_x} (first pred)")

    ax.set_xlabel("Decoding step t", fontsize=11)
    ax.set_ylabel("Probability", fontsize=11)
    ax.set_title(
        f"Bar {bar_x}: how predictions evolve as more bars are generated\n"
        f"Ground truth: {CHORD_CLASSES_ALL[true_cls]}   "
        f"AR output: {CHORD_CLASSES_ALL[ar_preds[bar_x]]}",
        fontsize=11
    )
    ax.set_xticks(steps)
    ax.set_xticklabels([str(t) for t in steps], fontsize=7)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


# ── plot 3: summary strip — top prediction per bar at each step ───────────────

def plot_prediction_strip(logits_evolution, target_seq, ar_preds,
                           out_path="prediction_evolution_strip.png"):
    """
    Grid of size (MAX_LEN, MAX_LEN):
      cell (x, t) shows the top-1 predicted chord for bar x at step t.
      Green = matches ground truth. Red = wrong. Grey = not yet reached.
    """
    MAX_LEN = logits_evolution.shape[0]
    fig, ax = plt.subplots(figsize=(18, 14))

    for x in range(MAX_LEN):
        true_cls = target_seq[x]
        for t in range(x, MAX_LEN):
            row = logits_evolution[t, x, :]
            if np.isnan(row).all():
                continue
            pred_cls = int(np.argmax(row))
            color = "#a8e6cf" if pred_cls == true_cls else "#ffaaa5"
            ax.add_patch(plt.Rectangle((t - 0.5, x - 0.5), 1, 1,
                                        color=color, lw=0))
            label = str(CHORD_CLASSES_ALL[pred_cls])
            # Abbreviate for readability
            label = label.replace(":maj", "M").replace(":min", "m") \
                         .replace(":maj7", "M7").replace(":min7", "m7")
            ax.text(t, x, label, ha="center", va="center", fontsize=4.5,
                    color="black")

    ax.set_xlim(-0.5, MAX_LEN - 0.5)
    ax.set_ylim(-0.5, MAX_LEN - 0.5)
    ax.invert_yaxis()
    ax.set_xticks(range(MAX_LEN))
    ax.set_xticklabels([str(t) for t in range(MAX_LEN)], fontsize=6, rotation=90)
    ax.set_yticks(range(MAX_LEN))
    ytick_labels = [f"{x}: {CHORD_CLASSES_ALL[target_seq[x]]}" for x in range(MAX_LEN)]
    ax.set_yticklabels(ytick_labels, fontsize=6)
    ax.set_xlabel("Decoding step t", fontsize=10)
    ax.set_ylabel("Bar x (ground truth)", fontsize=10)
    ax.set_title(
        "Top-1 prediction for bar x at each step t ≥ x\n"
        "Green = correct, Red = wrong",
        fontsize=11
    )
    # Diagonal
    ax.plot(range(MAX_LEN), range(MAX_LEN), color="black", lw=1.5,
            ls="--", alpha=0.5, label="First prediction (diagonal)")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


# ── latent distance to ground truth ──────────────────────────────────────────

def collect_gt_distances(model: TransformerModel,
                         melody: np.ndarray,
                         target: np.ndarray) -> np.ndarray:
    """
    For one window (32-bar context), run AR decoding and compute
    ||h_t − emb(gt_chord_t)|| for each of the 33 AR steps.

    Returns
    -------
    dists : np.ndarray  shape (MAX_LEN,)
    """
    model.eval()
    mel_t = torch.tensor(melody, dtype=torch.float32).to(DEVICE)   # (1, MEMORY, F)
    tgt_cls = target.squeeze()[:MAX_LEN]                            # (MAX_LEN,) int indices
    d_model = model.d_model
    dists = np.zeros(MAX_LEN, dtype=np.float32)

    with torch.no_grad():
        # GT chord embeddings, shape (MAX_LEN, d_model)
        gt_indices = torch.tensor(tgt_cls, dtype=torch.long, device=DEVICE)
        gt_embs = model.embedding_output(gt_indices).cpu().numpy()  # (MAX_LEN, d_model)

        enc_in = mel_t
        memory_enc = model.encoder(
            model.feature_to_embedding(
                torch.where(enc_in > 10, enc_in % 12, enc_in)
            ) + model.pos_encoder
        )

        tgt_emb = torch.zeros(1, 1, d_model, device=DEVICE)

        for t in range(MAX_LEN):
            out = model.decoder(tgt=tgt_emb, memory=memory_enc)
            h_t = out[0, -1, :].cpu().numpy()                      # (d_model,)
            dists[t] = np.linalg.norm(h_t - gt_embs[t])

            logits_t = model.fc_out(out[0, -1, :])
            probs_t = torch.softmax(logits_t, dim=-1)
            new_emb = (probs_t @ model.embedding_output.weight).unsqueeze(0).unsqueeze(0)
            tgt_emb = torch.cat([tgt_emb, new_emb], dim=1)

    return dists


def plot_latent_distance_heatmap(dist_matrix: np.ndarray, n_songs: int,
                                  n_sequences: int,
                                  out_path: str = "latent_distance.png"):
    """
    Plot the 33×33 heatmap of mean ||h_t − emb(gt_t)|| distances.

    Rows    = window position in song (0 = early, 32 = late), normalised to 33 bins
    Columns = AR decoding step t (0..32)
    """
    T = dist_matrix.shape[1]  # 33

    fig, axes = plt.subplots(1, 2, figsize=(18, 7),
                             gridspec_kw={"width_ratios": [1.1, 1]})

    # ── left: 33×33 heatmap ───────────────────────────────────────────────────
    ax = axes[0]
    im = ax.imshow(dist_matrix, aspect="auto", cmap="viridis_r", origin="upper")
    plt.colorbar(im, ax=ax, label="Mean ||h_t − emb(gt_chord_t)||")
    ticks = list(range(0, T, 4)) + [T - 1]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks], fontsize=8)
    ax.set_yticklabels([f"{r}" for r in ticks], fontsize=8)
    ax.set_xlabel("AR decoding step t (bar within window)", fontsize=11)
    ax.set_ylabel("Window position in song (early → late)", fontsize=11)
    ax.set_title(
        f"||h_t − emb(gt_chord_t)|| — {n_songs} songs, 33 windows each from middle\n"
        "Rows = consecutive windows (middle of song)   Cols = AR step within window",
        fontsize=10
    )

    # ── right: column means — distance per AR step, averaged over window pos ──
    ax2 = axes[1]
    col_mean = dist_matrix.mean(axis=0)   # (33,) — average over window positions
    ax2.plot(range(T), col_mean, marker="o", markersize=4, lw=1.8, color="tab:orange")
    ax2.fill_between(range(T), col_mean, alpha=0.15, color="tab:orange")
    ax2.set_xlabel("AR decoding step t", fontsize=11)
    ax2.set_ylabel("Mean distance to GT embedding", fontsize=11)
    ax2.set_title("Mean across window positions\n(how distance to GT evolves within a window)", fontsize=11)
    ax2.set_xticks(range(0, T, 4))
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Latent distance to ground truth chord embedding — sliding window view",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


def run_latent_distance_aggregate(model, song_ids, out_dir="AR", start_frac=0.5):
    """
    For each song, pick a random start in the middle section and take
    exactly MAX_LEN=33 consecutive windows from there.
    Row i of the heatmap = window (start + i), averaged over all songs.
    """
    T = MAX_LEN  # 33 rows, 33 cols
    dist_acc  = np.zeros((T, T), dtype=np.float64)
    song_count = 0

    rng = np.random.default_rng(42)

    for song_id in song_ids:
        npz_path = f"data/pop/melody_chords/{song_id:03d}.npz"
        try:
            melody, target = break_down_one_song_into_sequences(npz_path, test=True)
        except Exception as e:
            print(f"  Skipping song {song_id}: {e}")
            continue

        N = len(melody)
        if N < T:
            print(f"  Skipping song {song_id}: only {N} windows (need {T})")
            continue

        # Pick random start in the middle half [N//4, 3N//4)
        lo = N // 4
        hi = max(lo + 1, 3 * N // 4 - T)
        start_idx = int(rng.integers(lo, hi + 1))

        song_dist = np.zeros((T, T), dtype=np.float64)
        for row, seq_idx in enumerate(range(start_idx, start_idx + T)):
            mel = melody[seq_idx: seq_idx + 1]
            tgt = target[seq_idx: seq_idx + 1]
            try:
                dists = collect_gt_distances(model, mel, tgt)   # (T,)
                song_dist[row] = dists
            except Exception as e:
                print(f"  Error song {song_id} seq {seq_idx}: {e}")

        dist_acc += song_dist
        song_count += 1
        print(f"Song {song_id:03d} done  (start={start_idx}/{N}, songs so far: {song_count})")

    dist_matrix = (dist_acc / max(song_count, 1)).astype(np.float32)

    n_songs = len(song_ids)
    out_path = os.path.join(out_dir, f"agg{n_songs}s_latent_gt_distance.png")
    plot_latent_distance_heatmap(dist_matrix, n_songs=n_songs,
                                  n_sequences=song_count * T, out_path=out_path)
    print(f"Aggregated over {song_count} songs ({song_count * T} windows total).")


# ── helpers ───────────────────────────────────────────────────────────────────

def _softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualise how transformer predictions evolve bar by bar")
    parser.add_argument("--song",      type=int, default=707,   help="Song number")
    parser.add_argument("--seq",       type=int, default=-1,    help="Sequence index (-1 = densest)")
    parser.add_argument("--bar",       type=int, default=-1,    help="Bar to plot top-k evolution for (-1 = most interesting)")
    parser.add_argument("--topk",      type=int, default=5,     help="Top-k competitors to show")
    parser.add_argument("--outdir",    type=str, default="AR",  help="Output directory")
    parser.add_argument("--latent",    action="store_true",     help="Run latent distance heatmap (aggregate)")
    parser.add_argument("--n_songs",    type=int,   default=50,  help="Songs to aggregate over (--latent mode)")
    parser.add_argument("--start_frac", type=float, default=0.5, help="Fraction into song to start windows from (0=all, 0.5=middle)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("Loading model…")
    model = TransformerModel(INPUT_DIM, NUM_CLASSES_ALL).to(DEVICE)
    ckpt = torch.load("checkpoints/transformer_model.pth", map_location=DEVICE)
    model.load_state_dict(ckpt)

    if args.latent:
        all_songs = sorted([
            int(f.stem) for f in
            __import__("pathlib").Path("data/pop/melody_chords").glob("*.npz")
        ])
        song_ids = all_songs[:args.n_songs]
        run_latent_distance_aggregate(model, song_ids, out_dir=args.outdir,
                                      start_frac=args.start_frac)
        return

    npz_path = f"data/pop/melody_chords/{args.song:03d}.npz"
    print(f"Loading {npz_path}…")
    melody, target = break_down_one_song_into_sequences(npz_path, test=True)

    if args.seq == -1:
        note_counts = [
            sum(1 for beat in mel if any(n > 10 for n in beat[1:]))
            for mel in melody
        ]
        seq_idx = int(np.argmax(note_counts))
        print(f"Auto-selected sequence {seq_idx} ({note_counts[seq_idx]}/{MEMORY} beats with notes)")
    else:
        seq_idx = args.seq

    mel = melody[seq_idx: seq_idx + 1]   # (1, MEMORY, F)
    tgt = target[seq_idx: seq_idx + 1]   # (1, MAX_LEN, 1)

    print("Running evolution collection…")
    logits_ev, ar_preds, target_seq = collect_evolution(model, mel, tgt)

    print(f"Ground truth  : {[str(CHORD_CLASSES_ALL[c]) for c in target_seq]}")
    print(f"AR predictions: {[str(CHORD_CLASSES_ALL[c]) for c in ar_preds]}")
    acc = sum(p == g for p, g in zip(ar_preds, target_seq)) / MAX_LEN
    print(f"Accuracy: {acc:.1%}")

    prefix = os.path.join(args.outdir, f"song{args.song:03d}_seq{seq_idx}")

    plot_correct_prob_heatmap(logits_ev, target_seq, ar_preds,
                               out_path=prefix + "_prob_heatmap.png")

    plot_prediction_strip(logits_ev, target_seq, ar_preds,
                          out_path=prefix + "_strip.png")

    # Pick bar for top-k plot: most interesting = where AR got it wrong
    if args.bar == -1:
        wrong = [x for x, (p, g) in enumerate(zip(ar_preds, target_seq)) if p != g]
        bar_x = wrong[0] if wrong else 0
        print(f"Plotting top-k evolution for bar {bar_x} (first wrong prediction)")
    else:
        bar_x = args.bar

    plot_bar_topk_evolution(logits_ev, target_seq, ar_preds, bar_x,
                             k=args.topk,
                             out_path=prefix + f"_bar{bar_x}_topk.png")

    print("Done.")


if __name__ == "__main__":
    main()
