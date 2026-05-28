"""
Aggregate latent distance to ground truth over 50 songs and save to report/figures/.

Output: report/figures/transformer_latent_distance.png

Run from repo root:
    python report/scripts/gen_fig_latent_distance.py

Pass --replot to skip re-running the model and just replot from the saved numpy cache.
"""

import os
import sys
import argparse
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch

from AR.ar_transformer import TransformerModel
from AR.prediction_evolution import run_latent_distance_aggregate, plot_latent_distance_heatmap
from utils.constants import DEVICE, INPUT_DIM, NUM_CLASSES_ALL

OUT_DIR    = "report/figures"
CACHE_PATH = "report/figures/transformer_latent_distance_cache.npy"
N_SONGS    = 50
MODEL_PATH = "checkpoints/transformer_model.pth"

os.makedirs(OUT_DIR, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replot", action="store_true",
                        help="Skip model inference; replot from cached dist_matrix.")
    args = parser.parse_args()

    if args.replot and os.path.exists(CACHE_PATH):
        print(f"Loading cached dist_matrix from {CACHE_PATH}")
        dist_matrix = np.load(CACHE_PATH)
    else:
        print("Loading model…")
        model = TransformerModel(INPUT_DIM, NUM_CLASSES_ALL).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

        all_songs = sorted([
            int(f.stem) for f in
            __import__("pathlib").Path("data/pop/melody_chords").glob("*.npz")
        ])
        song_ids = all_songs[:N_SONGS]

        dist_matrix = run_latent_distance_aggregate(model, song_ids, out_dir=OUT_DIR)

        # cache for fast re-plotting
        np.save(CACHE_PATH, dist_matrix)
        print(f"Cached dist_matrix → {CACHE_PATH}")

        # rename the auto-named output
        import glob, shutil
        outputs = glob.glob(os.path.join(OUT_DIR, "agg*latent_gt_distance.png"))
        if outputs:
            shutil.move(outputs[0], os.path.join(OUT_DIR, "transformer_latent_distance.png"))
            print(f"Renamed to {os.path.join(OUT_DIR, 'transformer_latent_distance.png')}")

    # always re-plot (applies any style changes to plot_latent_distance_heatmap)
    out_path = os.path.join(OUT_DIR, "transformer_latent_distance.png")
    plot_latent_distance_heatmap(dist_matrix, n_songs=N_SONGS,
                                  n_sequences=N_SONGS * dist_matrix.shape[0],
                                  out_path=out_path)
    print("Done.")


if __name__ == "__main__":
    main()
