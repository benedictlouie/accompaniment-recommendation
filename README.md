# Harmonic Accompaniment Recommendation with Melody Tracking

A real-time AI system that listens to a solo melody and generates a full accompaniment — drums, bass, piano, and guitar — beat by beat.

Try it live: **[accompaniment-recommendation.vercel.app](https://accompaniment-recommendation.vercel.app)**

---

## 🚀 System Architecture

The system operates as a three-stage pipeline designed for low-latency live performance.

### 1. Real-Time Transcription
Captures live audio via `sounddevice` and performs fundamental frequency (f0) estimation.
- **Core technology**: `librosa.pyin` + a custom `LivePyin` for robust pitch tracking.
- **Beat alignment**: audio is buffered and processed beat-by-beat so the melody is accurately quantised for downstream models.

### 2. Chord Estimation (the "Harmonic Brain")
Predicts the most likely chord sequence from the temporal melody context.
- **AR Transformer** — maps the past 32 beats of melody to the current chord ($M_{t-32:t-1} \rightarrow C_t$). 6-layer encoder/decoder, d_model=128.
- **CRF** — Conditional Random Field with a Krumhansl–Kessler key-profile emission model and a learned chord-transition prior.

### 3. Rhythm / Symphony Generation
Retrieves multi-instrument loops matched to the melody's groove pattern.
- **k-NN retrieval** — the 16-dim onset pattern of each bar is matched against the LPD-5 database using Manhattan distance.
- **Chord snapping** — retrieved interval patterns are transposed in real time to fit the predicted chord.
- **Synthesis** — locally via FluidSynth + soundfonts; in the web app via the Web Audio API.

---

## 🌐 Web App (Vercel)

The web app runs fully in the browser. Audio synthesis (piano, guitar, bass, drums) is handled client-side with the Web Audio API; the Python API handles only chord prediction and loop selection.

### Architecture changes for serverless

| Problem | Solution |
|---|---|
| `torch` (~200 MB) too large for Vercel | AR Transformer exported to ONNX; CRF MLP exported to `.npz` — both run with `numpy`/`onnxruntime` only |
| `nn.joblib` (192 MB) gitignored | Subsample 50 K examples → `data/lpd/nn_web.npz` (~3 MB); numpy brute-force search |
| Stateful engine (history between beats) | History is base64-encoded and round-tripped in every request body |
| Server-side audio (FluidSynth, pygame) | Audio moved entirely to the browser via Web Audio API |

### One-time preparation (run locally, requires the full dataset)

```bash
# 1. Export the AR Transformer to ONNX
python scripts/export_onnx.py
# → deploy/model.onnx + deploy/decoder_weights.npz

# 2. Export the CRF MLP weights and transition matrix (requires torch locally)
python scripts/export_crf_nn.py
# → deploy/crf_nn.npz
# → deploy/crf_transition.npy

# 3. Compress the k-NN data for deployment
python scripts/compress_nn.py
# → deploy/nn_web.npz  (~3 MB)

# 4. Commit the generated artifacts
git add deploy/
git commit -m "add ONNX model + CRF weights + compressed NN for Vercel"
```

### Deploy to Vercel

```bash
npx vercel          # first time — follow prompts to link project
npx vercel --prod   # subsequent deploys
```

### Local API dev

```bash
python api/index.py          # starts on http://localhost:5001
# then open public/piano.html (or vercel dev for full routing)
```

Pages:
- `/` — landing page
- `/piano` — real-time piano accompaniment
- `/harmoniser` — real-time harmoniser

---

## 🛠️ Local Desktop App

The original pygame-based desktop apps still work as before.

### Prerequisites
- Python 3.9+
- [FluidSynth](https://www.fluidsynth.org/)
- Soundfonts in `soundfonts/`
- LPD index files (`data/lpd/nn.joblib` + `data/lpd/loops.npz`) from `data/lpd/extract.py`

### Install

```bash
pip install -r requirements-dev.txt
```

### Run

```bash
# Desktop piano app (pygame UI + FluidSynth audio)
python accompaniment/piano_app.py

# Web API for local browser testing
python api/index.py        # → http://localhost:5001
# then open public/piano.html or public/harmoniser.html
```

---

## 📊 Training

### Datasets

| Dataset | Use |
|---|---|
| POP909 (900+ songs) | Chord prediction training |
| EWLD (Wikifonia) | Chord prediction training |
| WJazzD (Weimar Jazz) | Chord prediction training |
| LPD-5 (Lakh Pianoroll) | Rhythm/groove k-NN index |

### Reproduce training

```bash
# 1. Extract and prepare chord data
python data/pop/extract.py
python data/wjazzd/extract.py
python data/prepare_training_data.py

# 2. Train the AR Transformer
python AR/ar_transformer.py

# 3. Build the LPD groove index (needs lpd_5_cleansed/)
python data/lpd/extract.py

# 4. Monitor training
python -m tensorboard.main --logdir=runs --port=6006
```

---

## 📁 Key Files

```
api/
  index.py                  Vercel-compatible Flask API (stateless)
engines/
  transformer_engine.py     PyTorch AR Transformer (local training)
  onnx_engine.py            ONNX AR Transformer (Vercel inference)
  crf_engine.py             CRF engine (local, uses torch)
CRF/
  web_engine.py             Numpy-only CRF engine (Vercel inference, no torch)
accompaniment/
  accompaniment_system.py   Desktop accompaniment (FluidSynth)
  nn_web.py                 Web loop lookup (numpy, no sklearn)
  nn.py                     Original loop lookup (sklearn k-NN)
public/
  piano.html                Piano app (web)
  harmoniser.html           Harmoniser (web)
  index.html                Landing page
scripts/
  export_onnx.py            Export PyTorch → ONNX (run once before deploy)
  compress_nn.py            Compress nn.joblib → nn_web.npz (run once before deploy)
data/lpd/
  storage.py                load/dump API for nn.joblib + loops.npz
checkpoints/
  transformer_model.pth     Latest AR Transformer weights
deploy/
  model.onnx                ONNX AR Transformer (generated by scripts/export_onnx.py)
  decoder_weights.npz       Embedding + positional weights (generated)
  crf_nn.npz                CRF MLP weights (generated by scripts/export_crf_nn.py)
  crf_transition.npy        Chord transition matrix (generated by scripts/export_crf_nn.py)
  nn_web.npz                Compressed 50 K groove samples (generated by scripts/compress_nn.py)
```
