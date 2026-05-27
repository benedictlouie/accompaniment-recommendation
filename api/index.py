"""
Vercel-compatible Flask API — stateless chord prediction.

State (beat history) is encoded as base64 and round-tripped through the client.
No audio playback here; loops are returned as JSON arrays for the browser.

Local dev:  python api/index.py
Vercel:     vercel dev  (routes /api/* here automatically)
"""
import sys
import os
import base64
import traceback

import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── project root on path ──────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.constants import INPUT_DIM, MEMORY, NOTE_TO_MIDI, STEPS_PER_BEAT, BEATS_PER_BAR

app = Flask(__name__)
CORS(app, origins="*")

# ── lazy singletons (cold-start friendly) ────────────────────────────────────
_onnx_engine = None
_loop_lookup = None


def _engine():
    global _onnx_engine
    if _onnx_engine is None:
        from engines.onnx_engine import ONNXTransformerEngine
        _onnx_engine = ONNXTransformerEngine()
    return _onnx_engine


def _loops():
    global _loop_lookup
    if _loop_lookup is None:
        from accompaniment.nn_web import WebLoopLookup
        _loop_lookup = WebLoopLookup()
    return _loop_lookup


# ── history serialisation ─────────────────────────────────────────────────────

def _decode_history(h64: str) -> np.ndarray:
    """base64 → float32 [N, INPUT_DIM].  Empty string → zero-row array."""
    if not h64:
        return np.zeros((0, INPUT_DIM), dtype=np.float32)
    raw = base64.b64decode(h64)
    arr = np.frombuffer(raw, dtype=np.float32).copy()
    return arr.reshape(-1, INPUT_DIM)


def _encode_history(arr: np.ndarray) -> str:
    """float32 [N, INPUT_DIM] → base64 string."""
    return base64.b64encode(arr.astype(np.float32).tobytes()).decode()


# ── beat feature builder ──────────────────────────────────────────────────────

def _build_beat(notes_played, beat_start: float, step_duration: float,
                beat_index: int) -> np.ndarray:
    """Return 1-D float32 array of length INPUT_DIM for one beat."""
    grid = -np.ones(STEPS_PER_BEAT, dtype=np.float32)
    for step in range(STEPS_PER_BEAT):
        s = beat_start + step * step_duration
        e = s + step_duration
        active = [
            NOTE_TO_MIDI[note]
            for note, ns, ne in notes_played
            if note in NOTE_TO_MIDI and min(ne, e) > max(ns, s)
        ]
        if active:
            grid[step] = float(max(active))
    strong = np.array([1.0 if beat_index == 1 else 0.0], dtype=np.float32)
    return np.concatenate([strong, grid])          # [INPUT_DIM]


# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/predict-chord", methods=["POST"])
def predict_chord():
    data = request.get_json(force=True) or {}

    raw_notes   = data.get("notes", [])
    beat_start  = float(data.get("beat_start", 0.0))
    beat_index  = int(data.get("beat_index", 1))
    tempo       = int(data.get("tempo", 100))
    history_b64 = data.get("history", "")

    beat_dur   = 60.0 / tempo
    step_dur   = beat_dur / STEPS_PER_BEAT

    notes_played = [
        (n["note"], float(n["start"]), float(n["end"]))
        for n in raw_notes
        if n.get("note") and n["note"] not in ("quiet", "no pitch")
    ]

    # ── build & append this beat to history ──────────────────────────────────
    history  = _decode_history(history_b64)
    new_beat = _build_beat(notes_played, beat_start, step_dur, beat_index)

    if len(history) == 0:
        empty = np.concatenate([np.zeros(1), -np.ones(STEPS_PER_BEAT)]).astype(np.float32)
        history = np.tile(empty, MEMORY).reshape(MEMORY, INPUT_DIM)

    history = np.vstack([history, new_beat[np.newaxis, :]])[-MEMORY:]

    # ── chord prediction ──────────────────────────────────────────────────────
    chord = "N"
    try:
        chord = _engine().predict(history)
    except Exception:
        traceback.print_exc()

    # ── loop retrieval (bar start only) ───────────────────────────────────────
    loops = {}
    if beat_index == 1:
        last_bar = history[-BEATS_PER_BAR:, -STEPS_PER_BEAT:].flatten()
        try:
            loops = _loops().get_loops(last_bar)
        except Exception:
            traceback.print_exc()

    return jsonify({
        "chord":   chord,
        "duration": beat_dur,
        "history": _encode_history(history),
        "loops":   loops,
    })


@app.route("/api/reset", methods=["POST"])
def reset():
    return jsonify({"status": "reset", "history": ""})


# ── local dev entry-point ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("API server → http://localhost:5001")
    app.run(debug=False, host="0.0.0.0", port=5001)
