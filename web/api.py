"""
Flask API backend for the piano app and harmoniser web UIs.
Run from the project root:  python web/api.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify
from flask_cors import CORS
from engines.factory import create_engine
from accompaniment.accompaniment_system import AccompanimentSystem

app = Flask(__name__)
CORS(app)

# engines[app_name][engine_type] -> engine instance
_engines: dict = {}
# one AccompanimentSystem per app (shared across engine types)
_accompaniment: dict = {}

VALID_ENGINE_TYPES = {"transformer", "crf"}


def _get_engine(app_name: str, engine_type: str, tempo: int):
    engine_type = engine_type if engine_type in VALID_ENGINE_TYPES else "transformer"
    key = (app_name, engine_type)
    if key not in _engines:
        _engines[key] = create_engine(engine_type, tempo)
    else:
        _engines[key].set_tempo(tempo)
    return _engines[key]


def _get_accompaniment(app_name: str) -> AccompanimentSystem:
    if app_name not in _accompaniment:
        _accompaniment[app_name] = AccompanimentSystem()
    return _accompaniment[app_name]


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/predict-chord", methods=["POST"])
def predict_chord():
    """
    Body JSON:
      {
        "app":        "piano" | "harmoniser",
        "notes":      [{"note": "C4", "start": 1234.56, "end": 1234.90}, ...],
        "beat_start": 1234.56,   // Unix seconds (performance.now()/1000 from JS)
        "beat_index": 1,         // 1-4
        "tempo":      100
      }
    Response:
      { "chord": "C:maj", "duration": 0.6 }
    """
    data = request.get_json(force=True)

    app_name    = data.get("app", "piano")
    engine_type = data.get("engine_type", "transformer")
    raw_notes   = data.get("notes", [])
    beat_start  = float(data.get("beat_start", 0.0))
    beat_index  = int(data.get("beat_index", 1))
    tempo       = int(data.get("tempo", 100))

    notes_played = [
        (n["note"], float(n["start"]), float(n["end"]))
        for n in raw_notes
        if n.get("note") and n["note"] != "quiet" and n["note"] != "no pitch"
    ]

    engine = _get_engine(app_name, engine_type, tempo)
    accompaniment = _get_accompaniment(app_name)

    try:
        chord, duration = engine.process_beat(notes_played, beat_start, beat_index)
        melody = engine.last_bar
        accompaniment.play_beat(melody, chord, tempo, beat_index)
        return jsonify({"chord": chord, "duration": duration})
    except Exception as exc:
        return jsonify({"error": str(exc), "chord": "N", "duration": 60 / tempo}), 500


@app.route("/api/reset", methods=["POST"])
def reset():
    """Reset engine history (useful when the user reloads the page)."""
    data        = request.get_json(force=True) or {}
    app_name    = data.get("app", "piano")
    engine_type = data.get("engine_type", "transformer")
    key = (app_name, engine_type)
    if key in _engines:
        _engines[key].reset()
    # clear cached loops so the next bar starts fresh
    if app_name in _accompaniment:
        acc = _accompaniment[app_name]
        acc.prev_drum_loop = None
        acc.guitar_loop = None
        acc.piano_loop = None
        acc.bass_loop = None
    return jsonify({"status": "reset"})


# ------------------------------------------------------------------

if __name__ == "__main__":
    print("Starting API server on http://localhost:5001")
    print("Keep this running while using the web apps.")
    app.run(debug=False, host="0.0.0.0", port=5001)
