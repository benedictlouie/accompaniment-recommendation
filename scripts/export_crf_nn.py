"""
Export CRF MLP weights + transition matrix for serverless deployment.
Requires torch (run locally before deploying to Vercel).

Outputs:
  deploy/crf_nn.npz         — SmallChordClassifier weights as float32 arrays
  deploy/crf_transition.npy — chord transition matrix (25, 25, 24)
"""
import sys, os, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from CRF.chord_melody_relation import SmallChordClassifier
from utils.constants import NUM_CLASSES

CHECKPOINT = "checkpoints/small_melody_chord_model.pth"
DEPLOY_DIR  = "deploy"

model = SmallChordClassifier(NUM_CLASSES)
model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
model.eval()

weights = {name: param.detach().numpy() for name, param in model.named_parameters()}
np.savez(os.path.join(DEPLOY_DIR, "crf_nn.npz"), **weights)
print(f"Saved {DEPLOY_DIR}/crf_nn.npz  ({len(weights)} weight arrays)")

shutil.copy("CRF/chord_transition_matrix.npy", os.path.join(DEPLOY_DIR, "crf_transition.npy"))
print(f"Copied CRF/chord_transition_matrix.npy → {DEPLOY_DIR}/crf_transition.npy")
