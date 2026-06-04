import json
import os
import torch
import numpy as np
from sklearn.decomposition import PCA

from AR.ar_transformer import TransformerModel
from utils.constants import (INPUT_DIM, NUM_CLASSES_ALL, NUM_QUALITIES_ALL,
                              QUALITIES_ALL, CHORD_CLASSES_ALL, DEVICE)

KEEP_QUALITIES = {'maj', 'min', 'maj7', 'min7', 'sus4', '7'}
KEEP_INDICES = [i for i, q in enumerate(QUALITIES_ALL) if q in KEEP_QUALITIES]

_COF_ROOTS = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]


def main():
    model_path = "checkpoints/transformer_model.pth"
    out_path = "report/figures/pca_3d.json"

    model = TransformerModel(INPUT_DIM, NUM_CLASSES_ALL).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    weights = model.fc_out.weight.detach().cpu().numpy()  # (169, hidden)

    # exclude the "N" (no-chord) class at the end — only 168 chord rows
    weights = weights[:168]

    pca = PCA(n_components=3)
    reduced = pca.fit_transform(weights)
    print("Explained variance:", pca.explained_variance_ratio_)

    # build filtered point list
    points = []
    for i in range(168):
        qi = i % NUM_QUALITIES_ALL
        if qi not in KEEP_INDICES:
            continue
        label = CHORD_CLASSES_ALL[i]
        root_idx = i // NUM_QUALITIES_ALL
        points.append({
            "label": label,
            "quality": QUALITIES_ALL[qi],
            "qi": qi,
            "root_idx": root_idx,
            "x": float(reduced[i, 0]),
            "y": float(reduced[i, 1]),
            "z": float(reduced[i, 2]),
        })

    # build cycle-of-fifths edges (per kept quality)
    edges = []
    for qi in KEEP_INDICES:
        ring = [r * NUM_QUALITIES_ALL + qi for r in _COF_ROOTS]
        for k in range(len(ring)):
            ia, ib = ring[k], ring[(k + 1) % len(ring)]
            edges.append({
                "qi": qi,
                "quality": QUALITIES_ALL[qi],
                "ax": float(reduced[ia, 0]), "ay": float(reduced[ia, 1]), "az": float(reduced[ia, 2]),
                "bx": float(reduced[ib, 0]), "by": float(reduced[ib, 1]), "bz": float(reduced[ib, 2]),
            })

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"points": points, "edges": edges}, f)
    print(f"Saved {len(points)} points and {len(edges)} edges → {out_path}")


if __name__ == "__main__":
    main()
