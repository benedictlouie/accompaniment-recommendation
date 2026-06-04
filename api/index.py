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

from utils.constants import INPUT_DIM, MEMORY, NOTE_TO_MIDI, STEPS_PER_BEAT, BEATS_PER_BAR, REVERSE_ROOT_MAP

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


# ── CRF engine (lazy) ────────────────────────────────────────────────────────

_crf_step = None

def _crf():
    global _crf_step
    if _crf_step is None:
        from CRF.web_engine import step as _s
        _crf_step = _s
    return _crf_step


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


def _decode_drum_loop(h64: str):
    """base64 → uint8 [16, 128] or None."""
    if not h64:
        return None
    try:
        raw = base64.b64decode(h64)
        arr = np.frombuffer(raw, dtype=np.uint8).copy()
        if arr.size != 16 * 128:
            return None
        return arr.reshape(16, 128)
    except Exception:
        return None


def _encode_drum_loop(arr: np.ndarray) -> str:
    """uint8 [16, 128] → base64 string."""
    return base64.b64encode(arr.astype(np.uint8).tobytes()).decode()


def _decode_crf_delta(h64: str, num_classes: int) -> np.ndarray | None:
    """base64 → float32 [NUM_CLASSES], or None if empty."""
    if not h64:
        return None
    raw = base64.b64decode(h64)
    return np.frombuffer(raw, dtype=np.float32).copy()


def _encode_crf_delta(arr: np.ndarray) -> str:
    return base64.b64encode(arr.astype(np.float32).tobytes()).decode()


def _decode_crf_bar_history(h64: str) -> np.ndarray:
    """base64 → float32 [N, 13], empty gives shape (0, 13)."""
    if not h64:
        return np.zeros((0, 13), dtype=np.float32)
    raw = base64.b64decode(h64)
    arr = np.frombuffer(raw, dtype=np.float32).copy()
    return arr.reshape(-1, 13)


def _encode_crf_bar_history(arr: np.ndarray) -> str:
    return base64.b64encode(arr.astype(np.float32).tobytes()).decode()


def _decode_crf_bar_pitch(h64: str) -> np.ndarray:
    """base64 → float32 [12] accumulated pitch durations, zeros if empty."""
    if not h64:
        return np.zeros(12, dtype=np.float32)
    raw = base64.b64decode(h64)
    return np.frombuffer(raw, dtype=np.float32).copy()


def _encode_crf_bar_pitch(arr: np.ndarray) -> str:
    return base64.b64encode(arr.astype(np.float32).tobytes()).decode()


def _notes_to_pitch_durations(notes_played) -> np.ndarray:
    """Convert [(note, start, end), ...] → float32[12] pitch-class durations."""
    pc_dur = np.zeros(12, dtype=np.float32)
    for note, start, end in notes_played:
        root = note[:-1]  # strip octave: "C#4" → "C#"
        if root in REVERSE_ROOT_MAP:
            pc_dur[REVERSE_ROOT_MAP[root]] += max(0.0, end - start)
    return pc_dur


def _bar_pitch_to_histogram(bar_pitch: np.ndarray) -> np.ndarray:
    """float32[12] pitch durations → float32[13] histogram (bin 0 = silence)."""
    bars = np.zeros(13, dtype=np.float32)
    total = bar_pitch.sum()
    if total < 1e-9:
        bars[0] = 1.0
    else:
        bars[1:] = bar_pitch / total
    return bars


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

    raw_notes      = data.get("notes", [])
    beat_start     = float(data.get("beat_start", 0.0))
    beat_index     = int(data.get("beat_index", 1))
    tempo          = int(data.get("tempo", 100))
    history_b64    = data.get("history", "")
    prev_drum_b64  = data.get("prev_drum_loop", "")
    engine_type    = data.get("engine", "transformer")

    beat_dur   = 60.0 / tempo
    step_dur   = beat_dur / STEPS_PER_BEAT

    notes_played = [
        (n["note"], float(n["start"]), float(n["end"]))
        for n in raw_notes
        if n.get("note") and n["note"] not in ("quiet", "no pitch")
    ]

    # ── CRF engine branch ─────────────────────────────────────────────────────
    if engine_type == "crf":
        crf_delta_b64    = data.get("crf_delta", "")
        crf_history_b64  = data.get("crf_bar_history", "")
        crf_pitch_b64    = data.get("crf_bar_pitch", "")
        crf_beat_count   = int(data.get("crf_beat_count", 0))
        crf_loop_hist_b64 = data.get("crf_loop_history", "")

        delta       = _decode_crf_delta(crf_delta_b64, 25)
        bar_history = _decode_crf_bar_history(crf_history_b64)
        bar_pitch   = _decode_crf_bar_pitch(crf_pitch_b64)

        # Build melody beat feature for loop retrieval (same as transformer path)
        new_beat = _build_beat(notes_played, beat_start, step_dur, beat_index)
        loop_history = _decode_history(crf_loop_hist_b64)
        if len(loop_history) == 0:
            empty = np.concatenate([np.zeros(1), -np.ones(STEPS_PER_BEAT)]).astype(np.float32)
            loop_history = np.tile(empty, MEMORY).reshape(MEMORY, INPUT_DIM)
        loop_history = np.vstack([loop_history, new_beat[np.newaxis, :]])[-MEMORY:]

        bar_pitch += _notes_to_pitch_durations(notes_played)
        crf_beat_count += 1

        chord = None  # no prediction until bar is complete
        if crf_beat_count >= BEATS_PER_BAR:
            bars = _bar_pitch_to_histogram(bar_pitch)
            try:
                chord, delta, bar_history = _crf()(bars, delta, bar_history)
            except Exception:
                traceback.print_exc()
            bar_pitch      = np.zeros(12, dtype=np.float32)
            crf_beat_count = 0

        # ── loop retrieval (bar start only, same logic as transformer) ─────────
        loops          = {}
        prev_drum_loop = _decode_drum_loop(prev_drum_b64)
        next_drum_b64  = prev_drum_b64

        if beat_index == 1:
            last_bar = loop_history[-BEATS_PER_BAR:, -STEPS_PER_BEAT:].flatten()
            try:
                candidate  = _loops().get_loops(last_bar)
                drum_arr   = np.array(candidate["drums"], dtype=np.uint8)
                drum_steps = np.nonzero(drum_arr)[0]
                if len(drum_steps) < 10:
                    if prev_drum_loop is not None:
                        drum_arr = prev_drum_loop
                        candidate["drums"] = prev_drum_loop.tolist()
                    else:
                        candidate["drums"] = np.zeros((16, 128), dtype=np.uint8).tolist()
                        drum_arr = np.zeros((16, 128), dtype=np.uint8)
                else:
                    if prev_drum_loop is not None:
                        a = (prev_drum_loop > 0).flatten()
                        b = (drum_arr > 0).flatten()
                        union = np.logical_or(a, b).sum()
                        if union > 0 and 1.0 - np.logical_and(a, b).sum() / union > 0.75:
                            drum_arr = prev_drum_loop
                            candidate["drums"] = prev_drum_loop.tolist()
                    next_drum_b64 = _encode_drum_loop(drum_arr)
                loops = candidate
            except Exception:
                traceback.print_exc()

        return jsonify({
            "chord":            chord,
            "duration":         beat_dur,
            "history":          history_b64,
            "loops":            loops,
            "prev_drum_loop":   next_drum_b64,
            "crf_delta":        _encode_crf_delta(delta) if delta is not None else "",
            "crf_bar_history":  _encode_crf_bar_history(bar_history),
            "crf_bar_pitch":    _encode_crf_bar_pitch(bar_pitch),
            "crf_beat_count":   crf_beat_count,
            "crf_loop_history": _encode_history(loop_history),
        })

    # ── Transformer engine branch (default) ───────────────────────────────────
    history  = _decode_history(history_b64)

    # Early-fire peek: predict using history[-n_real:] + latency_comp padding rows.
    # Does NOT update history — client calls again at the beat boundary for that.
    early_fire   = bool(data.get("early_fire", False))
    latency_comp = int(data.get("latency_compensation", 0))
    if early_fire and latency_comp > 0:
        if len(history) == 0:
            empty_row = np.concatenate([np.zeros(1), -np.ones(STEPS_PER_BEAT)]).astype(np.float32)
            history = np.tile(empty_row, MEMORY).reshape(MEMORY, INPUT_DIM)
        n_real   = max(0, MEMORY - latency_comp)
        real     = history[-n_real:]
        last_row = real[-1].copy() if len(real) > 0 else np.concatenate([np.zeros(1), -np.ones(STEPS_PER_BEAT)]).astype(np.float32)
        last_row[0] = 0.0
        # Each padding row represents one future beat starting at beat_index.
        # Set bar_start_flag=1 for whichever rows land on beat 1.
        padding_rows = []
        for i in range(latency_comp):
            row = last_row.copy()
            row[0] = 1.0 if ((beat_index - 1 + i) % BEATS_PER_BAR == 0) else 0.0
            padding_rows.append(row)
        padding  = np.array(padding_rows, dtype=np.float32)
        peek_h   = np.vstack([real, padding]) if len(real) > 0 else padding
        chord    = "N"
        try:
            chord = _engine().predict(peek_h.astype(np.float32))
        except Exception:
            traceback.print_exc()
        return jsonify({"chord": chord})

    new_beat = _build_beat(notes_played, beat_start, step_dur, beat_index)

    if len(history) == 0:
        empty = np.concatenate([np.zeros(1), -np.ones(STEPS_PER_BEAT)]).astype(np.float32)
        history = np.tile(empty, MEMORY).reshape(MEMORY, INPUT_DIM)

    history = np.vstack([history, new_beat[np.newaxis, :]])[-MEMORY:]

    chord = "N"
    try:
        chord = _engine().predict(history)
    except Exception:
        traceback.print_exc()

    # ── loop retrieval (bar start only) ───────────────────────────────────────
    loops          = {}
    prev_drum_loop = _decode_drum_loop(prev_drum_b64)
    next_drum_b64  = prev_drum_b64

    if beat_index == 1:
        last_bar = history[-BEATS_PER_BAR:, -STEPS_PER_BEAT:].flatten()
        try:
            candidate  = _loops().get_loops(last_bar)
            drum_arr   = np.array(candidate["drums"], dtype=np.uint8)
            drum_steps = np.nonzero(drum_arr)[0]

            if len(drum_steps) < 10:
                if prev_drum_loop is not None:
                    drum_arr = prev_drum_loop
                    candidate["drums"] = prev_drum_loop.tolist()
                else:
                    candidate["drums"] = np.zeros((16, 128), dtype=np.uint8).tolist()
                    drum_arr = np.zeros((16, 128), dtype=np.uint8)
            else:
                if prev_drum_loop is not None:
                    a = (prev_drum_loop > 0).flatten()
                    b = (drum_arr > 0).flatten()
                    union = np.logical_or(a, b).sum()
                    if union > 0 and 1.0 - np.logical_and(a, b).sum() / union > 0.75:
                        drum_arr = prev_drum_loop
                        candidate["drums"] = prev_drum_loop.tolist()
                next_drum_b64 = _encode_drum_loop(drum_arr)

            loops = candidate
        except Exception:
            traceback.print_exc()

    return jsonify({
        "chord":          chord,
        "duration":       beat_dur,
        "history":        _encode_history(history),
        "loops":          loops,
        "prev_drum_loop": next_drum_b64,
    })


@app.route("/api/reset", methods=["POST"])
def reset():
    return jsonify({"status": "reset", "history": ""})


# ── local dev entry-point ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("API server → http://localhost:5001")
    app.run(debug=False, host="0.0.0.0", port=5001)
