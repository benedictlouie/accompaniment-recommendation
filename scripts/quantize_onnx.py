"""
One-time script: INT8 dynamic quantization of the exported ONNX model.

Run from the project root AFTER export_onnx.py:
    python scripts/quantize_onnx.py

Produces:
    deploy/model_int8.onnx   – INT8 weights, ~4x smaller, 2-4x faster on CPU

The onnx_engine.py will automatically prefer model_int8.onnx when present.
"""
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from onnxruntime.quantization import quantize_dynamic, QuantType, shape_inference

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_PATH   = os.path.join(ROOT, "deploy", "model.onnx")
OUT_PATH  = os.path.join(ROOT, "deploy", "model_int8.onnx")

if not os.path.exists(IN_PATH):
    print(f"ERROR: {IN_PATH} not found. Run scripts/export_onnx.py first.")
    sys.exit(1)

# ORT requires a pre-processing pass to fix shape inference issues (e.g. tied
# weights whose transpose introduces a shape ambiguity).
with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
    PREP_PATH = f.name

import onnx

# Strip stale intermediate shape annotations — these conflict with ORT's own
# shape inference pass inside quantize_dynamic, causing an InferenceError.
print(f"Stripping shape annotations from {IN_PATH} …")
m = onnx.load(IN_PATH)
del m.graph.value_info[:]
onnx.save(m, PREP_PATH)

print(f"Quantizing …")
quantize_dynamic(
    model_input  = PREP_PATH,
    model_output = OUT_PATH,
    weight_type  = QuantType.QInt8,
    per_channel  = False,
    reduce_range = False,
)
os.unlink(PREP_PATH)

in_mb  = os.path.getsize(IN_PATH)  / 1e6
out_mb = os.path.getsize(OUT_PATH) / 1e6
print(f"✓ {IN_PATH}  →  {OUT_PATH}")
print(f"  {in_mb:.1f} MB  →  {out_mb:.1f} MB  ({100*(1-out_mb/in_mb):.0f}% smaller)")
print("\nDone. The onnx_engine will use model_int8.onnx automatically.")
