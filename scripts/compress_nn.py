"""
One-time script: subsample nn.joblib + loops.npz into a small web-deployable file.

Run from the project root BEFORE deploying to Vercel:
    python scripts/compress_nn.py

Requires:
    data/lpd/nn.joblib    (192 MB, gitignored)
    data/lpd/loops.npz    (28 MB)

Produces:
    data/lpd/nn_web.npz   (< 5 MB, committed to git)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from data.lpd import storage as lpd_storage

N_SAMPLES = 50_000   # keep 50 K of the ~3 M examples — plenty for good retrieval
SEED      = 42
OUT_PATH  = "deploy/nn_web.npz"

print("Loading nn.joblib and loops.npz …  (this may take a moment)")
nn_model, drums  = lpd_storage.load("drums")
_,        pianos  = lpd_storage.load("piano")
_,        guitars = lpd_storage.load("guitar")
_,        basses  = lpd_storage.load("bass")

features = nn_model._fit_X          # [N, 16] – the groove feature matrix
N        = len(features)
print(f"  Total examples : {N:,}")
print(f"  drums  : {drums.shape}")
print(f"  piano  : {pianos.shape}")
print(f"  guitar : {guitars.shape}")
print(f"  bass   : {basses.shape}")

rng = np.random.default_rng(SEED)
idx = rng.choice(N, size=min(N_SAMPLES, N), replace=False)
idx.sort()

features_s = features[idx].astype(np.float32)
drums_s    = drums[idx]
pianos_s   = pianos[idx]
guitars_s  = guitars[idx]
basses_s   = basses[idx]

print(f"\nSubsampled → {len(idx):,} examples")
print(f"  features : {features_s.shape}  dtype={features_s.dtype}")
print(f"  drums    : {drums_s.shape}     dtype={drums_s.dtype}")
print(f"  piano    : {pianos_s.shape}    dtype={pianos_s.dtype}")
print(f"  guitar   : {guitars_s.shape}   dtype={guitars_s.dtype}")
print(f"  bass     : {basses_s.shape}    dtype={basses_s.dtype}")

np.savez_compressed(
    OUT_PATH,
    features = features_s,
    drums    = drums_s,
    piano    = pianos_s,
    guitar   = guitars_s,
    bass     = basses_s,
)

size_mb = os.path.getsize(OUT_PATH) / 1e6
print(f"\n✓ Saved → {OUT_PATH}  ({size_mb:.1f} MB)")
print("Done. You can now deploy to Vercel.")
