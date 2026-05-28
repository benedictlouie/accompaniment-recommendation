"""
PCA of AR Transformer output weights, ported from AR/pca_visualise_classes.py.
Run from repo root:
    python report/scripts/gen_fig_transformer_pca_nn.py
Outputs:
    report/figures/transformer_pca.png       — PC1 vs Phase(PC2,PC3), coloured by radius
    report/figures/transformer_pca_pc23.png  — PC2 vs PC3 with nearest-neighbour lines
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

from AR.ar_transformer import TransformerModel
from utils.constants import INPUT_DIM, NUM_CLASSES_ALL, NUM_QUALITIES_ALL, QUALITIES_ALL, CHORD_CLASSES_ALL, DEVICE

MODEL_PATH = "checkpoints/transformer_model.pth"
OUT_DIR    = "report/figures"

os.makedirs(OUT_DIR, exist_ok=True)

model = TransformerModel(INPUT_DIM, NUM_CLASSES_ALL).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

W = model.fc_out.weight.detach().cpu().numpy()   # [169, 128]
pca = PCA(n_components=6)
coords = pca.fit_transform(W)                    # [169, 6]
print(f"Explained variance (PC1–PC6): {pca.explained_variance_ratio_}")

qualities = np.arange(len(coords)) % NUM_QUALITIES_ALL
cmap = matplotlib.colormaps.get_cmap("tab20").resampled(NUM_QUALITIES_ALL)

# Cycle-of-fifths root order (semitone indices)
COF_ROOTS = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]

def _draw_cof_lines(ax, all_coords_2d, alpha=0.45, lw=1.2, skip_outliers=False):
    """
    Draw cycle-of-fifths ring connections for each quality group.
    skip_outliers: skip edges whose length exceeds median + 3*MAD across all edges
                   (removes phase-wrap-around long lines without isolating any node).
    """
    # Per-quality: skip edges above the natural length gap, but never isolate a node.
    # Skipping is greedy longest-first; a node is never left with zero connections.
    skip_outliers_threshold = np.inf
    if skip_outliers:
        all_lens = []
        for q in range(NUM_QUALITIES_ALL):
            ring = [r * NUM_QUALITIES_ALL + q for r in COF_ROOTS]
            for i in range(len(ring)):
                a = ring[i]; b = ring[(i + 1) % len(ring)]
                all_lens.append(np.linalg.norm(all_coords_2d[a] - all_coords_2d[b]))
        desc = np.sort(all_lens)[::-1]
        gap_idx = int(np.argmax(desc[:-1] - desc[1:]))
        skip_outliers_threshold = (desc[gap_idx] + desc[gap_idx + 1]) / 2

    for q in range(NUM_QUALITIES_ALL):
        color = cmap(q)
        ring = [r * NUM_QUALITIES_ALL + q for r in COF_ROOTS]
        n = len(ring)
        lens = [np.linalg.norm(all_coords_2d[ring[i]] - all_coords_2d[ring[(i+1)%n]])
                for i in range(n)]
        # Greedily skip long edges, protecting each node from isolation
        degree = [2] * n
        skip_set = set()
        for i in np.argsort(lens)[::-1]:
            if lens[i] <= skip_outliers_threshold:
                break
            ai, bi = i, (i + 1) % n
            if degree[ai] > 1 and degree[bi] > 1:
                skip_set.add(i)
                degree[ai] -= 1
                degree[bi] -= 1
        for i in range(n):
            if i in skip_set:
                continue
            pa = all_coords_2d[ring[i]]
            pb = all_coords_2d[ring[(i + 1) % n]]
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]],
                    color=color, alpha=alpha, linewidth=lw, zorder=2)


# ── Figure 1: PC1 vs Phase(PC2, PC3)  (replicates visualize_class_weights_2d) ──

pc1    = coords[:, 0]
pc2    = coords[:, 1]
pc3    = coords[:, 2]
phase  = np.arctan2(pc3, pc2)
radius = np.sqrt(pc2**2 + pc3**2)

fig, ax = plt.subplots(figsize=(12, 8))

for q in range(NUM_QUALITIES_ALL):
    idxs = np.where(qualities == q)[0]
    xg = pc1[idxs]
    yg = phase[idxs]
    rg = radius[idxs]

    sc = ax.scatter(xg, yg, c=rg, cmap="viridis",
                    vmin=radius.min(), vmax=radius.max(),
                    s=35, zorder=3)

    for i in idxs:
        ax.text(pc1[i], phase[i], CHORD_CLASSES_ALL[i],
                fontsize=7.5, ha="center", va="bottom", color=cmap(q))

# draw cycle-of-fifths edges, skipping longest per quality (removes phase-crossing long lines)
_draw_cof_lines(ax, np.stack([pc1, phase], axis=1)[:168], alpha=0.5, lw=1.8,
                skip_outliers=True)

cbar = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.04)
cbar.set_label(r"Radius $\sqrt{PC_2^2+PC_3^2}$", fontsize=10)
ax.legend(
    handles=[mpatches.Patch(color=cmap(q), label=QUALITIES_ALL[q])
             for q in range(NUM_QUALITIES_ALL)],
    bbox_to_anchor=(1.12, 1), loc="upper left",
    fontsize=7, ncol=1, framealpha=0.8, borderaxespad=0,
)
ax.set_xlabel("PC 1", fontsize=12)
ax.set_ylabel(r"Phase $\theta = \arctan2(PC_3,\,PC_2)$  (rad)", fontsize=12)
ax.set_title(
    "PCA of AR Transformer Output Weights — PC$_1$ vs Phase(PC$_2$, PC$_3$)\n"
    "(colour = radius in PC$_2$--PC$_3$ plane; lines = cycle-of-fifths within quality)",
    fontsize=11,
)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "transformer_pca.png"), dpi=150, bbox_inches="tight")
print(f"Saved → {os.path.join(OUT_DIR, 'transformer_pca.png')}")
plt.close(fig)


# ── Figure 2: PC2 vs PC3 with nearest-neighbour lines ─────────────────────────

fig, ax = plt.subplots(figsize=(11, 8))
for q in range(NUM_QUALITIES_ALL):
    mask = np.where(qualities == q)[0]
    gc = coords[mask][:, 1:3]
    ax.scatter(gc[:, 0], gc[:, 1], color=cmap(q), s=55, zorder=3)
    for i, idx in enumerate(mask):
        ax.annotate(
            CHORD_CLASSES_ALL[idx],
            (gc[i, 0], gc[i, 1]),
            fontsize=5.0, ha="center", va="bottom",
            color=cmap(q), xytext=(0, 3), textcoords="offset points",
        )

# cycle-of-fifths ring connections in the PC2-PC3 plane
_draw_cof_lines(ax, coords[:168, 1:3], alpha=0.4, lw=0.9, skip_outliers=False)
ax.set_xlabel("PC 2", fontsize=12)
ax.set_ylabel("PC 3", fontsize=12)
ax.set_title(
    "PCA of AR Transformer Output Layer Weights — PC2 vs PC3\n"
    "(lines connect chords in cycle-of-fifths order within each quality group)",
    fontsize=12,
)
ax.legend(
    handles=[mpatches.Patch(color=cmap(q), label=QUALITIES_ALL[q])
             for q in range(NUM_QUALITIES_ALL)],
    loc="upper right", fontsize=7, ncol=2, framealpha=0.8,
)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "transformer_pca_pc23.png"), dpi=150, bbox_inches="tight")
print(f"Saved → {os.path.join(OUT_DIR, 'transformer_pca_pc23.png')}")
plt.close(fig)
