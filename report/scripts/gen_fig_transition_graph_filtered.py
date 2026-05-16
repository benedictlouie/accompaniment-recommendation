"""
Chord transition graph showing only maj, min, and 7 chords.
Run from repo root:
    python report_final/scripts/gen_fig_transition_graph_filtered.py
Output: report_final/figures/transition_graph.png  (overwrites)
"""

import os, sys, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from data.chord2vec import load_embeddings, compute_pca_3d

TRANSITIONS_PATH = "data/transition_graph.json"
EMBED_PATH       = "data/token_embeddings_32d.json"
OUT_PATH         = "report_final/figures/transition_graph.png"

KEEP_QUALITIES = {"maj", "min", "7"}
MIN_WEIGHT = 50000

QUALITY_COLORS = {
    "maj": "firebrick",
    "min": "steelblue",
    "7":   "darkorange",
}

# PCA positions from Chord2Vec embeddings
tokens, labels, mod, embeddings = load_embeddings(EMBED_PATH)
coords = compute_pca_3d(embeddings)
pca_pos = {labels[i]: (coords[i][0], coords[i][1]) for i in range(len(labels))}

with open(TRANSITIONS_PATH) as f:
    transitions = json.load(f)

def get_quality(label):
    if ":" not in label:
        return None
    return label.split(":")[1]

G = nx.DiGraph()
node_quality = {}

for edge in transitions:
    w = edge["weight"]
    if w < MIN_WEIGHT:
        continue
    src = edge["source_label"]
    tgt = edge["target_label"]
    if src == tgt:
        continue
    src_q = get_quality(src)
    tgt_q = get_quality(tgt)
    if src_q not in KEEP_QUALITIES or tgt_q not in KEEP_QUALITIES:
        continue
    G.add_edge(src, tgt, weight=w)
    node_quality[src] = src_q
    node_quality[tgt] = tgt_q

print(f"Graph: {len(G.nodes)} nodes, {len(G.edges)} edges")

if len(G.nodes) == 0:
    raise ValueError("No edges passed filters — lower MIN_WEIGHT.")

pos = {n: pca_pos.get(n, (0.0, 0.0)) for n in G.nodes}

fig, ax = plt.subplots(figsize=(13, 10))

for q, color in QUALITY_COLORS.items():
    nodes = [n for n in G.nodes if node_quality.get(n) == q]
    nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=color,
                           node_size=260, ax=ax, alpha=0.9)

nx.draw_networkx_labels(G, pos, font_size=6, ax=ax, font_color="white", font_weight="bold")

weights = np.array([G[u][v]["weight"] for u, v in G.edges()])
widths  = 0.4 + 4.0 * (weights / weights.max())
edge_colors = [QUALITY_COLORS.get(node_quality.get(u, "maj"), "gray") for u, v in G.edges()]

nx.draw_networkx_edges(
    G, pos,
    width=widths,
    edge_color=edge_colors,
    arrows=True,
    arrowsize=12,
    connectionstyle="arc3,rad=0.08",
    alpha=0.7,
    ax=ax,
)

ax.set_title(
    f"Chord Transition Graph — maj, min, dom-7 only\n"
    f"(edges ≥ {MIN_WEIGHT:,} occurrences; width ∝ frequency)",
    fontsize=12,
)
ax.axis("off")
ax.legend(
    handles=[mpatches.Patch(color=c, label=q) for q, c in QUALITY_COLORS.items()],
    loc="lower right", fontsize=10, framealpha=0.85,
)

plt.tight_layout()
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT_PATH}")
