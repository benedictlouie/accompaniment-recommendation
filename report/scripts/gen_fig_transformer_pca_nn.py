"""
PCA of AR Transformer output weights with nearest-2 neighbour lines
drawn within each quality group (ported from AR/pca_visualise_classes.py).
Run from repo root:
    python report_final/scripts/gen_fig_transformer_pca_nn.py
Output: report_final/figures/transformer_pca.png  (overwrites)
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

from AR.ar_transformer import TransformerModel
from utils.constants import INPUT_DIM, NUM_CLASSES_ALL, NUM_QUALITIES_ALL, QUALITIES_ALL, CHORD_CLASSES_ALL, DEVICE

MODEL_PATH = "checkpoints/transformer_model.pth"
OUT_PATH   = "report_final/figures/transformer_pca.png"

model = TransformerModel(INPUT_DIM, NUM_CLASSES_ALL).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

W = model.fc_out.weight.detach().cpu().numpy()   # [169, 128]
pca = PCA(n_components=2)
coords = pca.fit_transform(W)                    # [169, 2]
print(f"Explained variance (PC1, PC2): {pca.explained_variance_ratio_}")

pc1, pc2 = coords[:, 0], coords[:, 1]
qualities = np.arange(len(pc1)) % NUM_QUALITIES_ALL

cmap = plt.cm.get_cmap("tab20", NUM_QUALITIES_ALL)
fig, ax = plt.subplots(figsize=(11, 8))

for q in range(NUM_QUALITIES_ALL):
    mask = np.where(qualities == q)[0]
    group_coords = coords[mask]   # [12, 2]

    # scatter + labels
    ax.scatter(group_coords[:, 0], group_coords[:, 1],
               color=cmap(q), s=55, zorder=3)
    for i, idx in enumerate(mask):
        ax.annotate(
            CHORD_CLASSES_ALL[idx],   # full label e.g. "C:maj"
            (group_coords[i, 0], group_coords[i, 1]),
            fontsize=5.0, ha="center", va="bottom",
            color=cmap(q), xytext=(0, 3), textcoords="offset points",
        )

    # nearest-2 lines within the same quality group (from pca_visualise_classes.py)
    for i, p in enumerate(group_coords):
        dists = np.linalg.norm(group_coords - p, axis=1)
        dists[i] = np.inf
        nearest = np.argsort(dists)[:2]
        for n in nearest:
            p2 = group_coords[n]
            ax.plot(
                [p[0], p2[0]],
                [p[1], p2[1]],
                color=cmap(q), alpha=0.4, linewidth=0.8, zorder=2,
            )

ax.set_xlabel("PC 1", fontsize=12)
ax.set_ylabel("PC 2", fontsize=12)
ax.set_title(
    "PCA of AR Transformer Output Layer Weights\n"
    "(lines connect 2 nearest neighbours within each quality group)",
    fontsize=12,
)
ax.legend(
    handles=[mpatches.Patch(color=cmap(q), label=QUALITIES_ALL[q]) for q in range(NUM_QUALITIES_ALL)],
    loc="upper right", fontsize=7, ncol=2, framealpha=0.8,
)
ax.grid(True, alpha=0.3)
plt.tight_layout()

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT_PATH}")
