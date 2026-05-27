"""
One-time script: export the AR Transformer to a single self-contained ONNX model.

Run from the project root BEFORE deploying to Vercel:
    python scripts/export_onnx.py

Produces:
    deploy/model.onnx           – full encoder + unrolled decoder (fixed I/O shapes)
    deploy/decoder_weights.npz  – kept for sampling (embedding matrix + temperature)

Strategy: wrap the full forward pass and trace it.  The autoregressive loop has
MAX_LEN=33 fixed iterations so torch.jit.trace unrolls it into a static graph
with no dynamic shapes — compatible with any onnxruntime version.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np

from AR.ar_transformer import TransformerModel, MAX_LEN
from utils.constants import INPUT_DIM, NUM_CLASSES_ALL, MEMORY

CHECKPOINT  = "checkpoints/transformer_model.pth"
MODEL_OUT   = "deploy/model.onnx"
WEIGHTS_OUT = "deploy/decoder_weights.npz"

os.makedirs("deploy", exist_ok=True)

# ── load model ────────────────────────────────────────────────────────────────
print(f"Loading {CHECKPOINT} …")
model = TransformerModel(INPUT_DIM, NUM_CLASSES_ALL)
state = torch.load(CHECKPOINT, map_location="cpu")
model.load_state_dict(state)
model.eval()
d_model = model.d_model
print(f"  d_model={d_model}  NUM_CLASSES={NUM_CLASSES_ALL}  MEMORY={MEMORY}  MAX_LEN={MAX_LEN}")

# ── full-inference wrapper ────────────────────────────────────────────────────
# Takes input_seq [1, MEMORY, INPUT_DIM] → logits [1, NUM_CLASSES_ALL]
# All shapes are fixed so torch.jit.trace unrolls the AR loop into a static graph.
class FullInferenceWrapper(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_seq):
        outputs = self.m(input_seq)          # [1, MAX_LEN, NUM_CLASSES]
        return outputs[:, -1, :]            # [1, NUM_CLASSES] – last step only

wrapper = FullInferenceWrapper(model)
wrapper.eval()
dummy = torch.zeros(1, MEMORY, INPUT_DIM)

# ── export to ONNX ────────────────────────────────────────────────────────────
# Pass the nn.Module directly (not JIT-traced).
# The dynamo exporter unrolls range(MAX_LEN) because MAX_LEN is a Python constant.
# All I/O shapes are fixed: [1,MEMORY,INPUT_DIM] → [1,NUM_CLASSES].
import onnx as _onnx

print("Exporting to ONNX …")
torch.onnx.export(
    wrapper,
    (dummy,),
    MODEL_OUT,
    input_names         = ["input_seq"],
    output_names        = ["logits"],
    opset_version       = 17,
    do_constant_folding = True,
)

# Merge external sidecar if the new exporter created one
m = _onnx.load(MODEL_OUT)
_onnx.save(m, MODEL_OUT, save_as_external_data=False)
for f in [MODEL_OUT + ".data"]:
    if os.path.exists(f): os.remove(f)

print(f"✓ Model    → {MODEL_OUT}  ({os.path.getsize(MODEL_OUT)/1e6:.1f} MB)")

# ── save weights for softmax sampling in Python ───────────────────────────────
# (the ONNX model returns raw logits; we sample in onnx_engine.py)
embedding_weight = model.embedding_output.weight.detach().numpy()  # [NUM_CLASSES, d_model]
pos_decoder      = model.pos_decoder.detach().numpy()               # kept for reference

np.savez_compressed(WEIGHTS_OUT,
                    embedding_weight=embedding_weight,
                    pos_decoder=pos_decoder)
print(f"✓ Weights  → {WEIGHTS_OUT}  ({os.path.getsize(WEIGHTS_OUT)/1e6:.2f} MB)")
print("\nDone. Next: python scripts/compress_nn.py")
