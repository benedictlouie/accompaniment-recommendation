# Harmonic Accompaniment Recommendation with Melody Tracking

A real-time AI-powered system that listens to a solo melody (vocal or instrumental) and generates a full "symphony" accompaniment in real-time, including drums, bass, piano, and guitar.

## 🚀 System Architecture

The system operates as a three-stage pipeline designed for low-latency live performance:

### 1. Real-Time Transcription
The system captures live audio via `sounddevice` and performs fundamental frequency (f0) estimation. 
- **Core Technology**: Uses `librosa.pyin` and a custom `LivePyin` implementation for robust pitch tracking.
- **Beat Alignment**: Audio is buffered and processed beat-by-beat to ensure the melody is accurately quantized for the downstream models.

### 2. Chord Estimation (The "Harmonic Brain")
Predicts the most likely chord sequence based on the temporal relationship between melody and harmony.
- **Models**: Supports multiple inference engines including **Conditional Random Fields (CRF)** and **Auto-Regressive Transformers**.
- **Sequence Mapping**: The system models the mapping from past melody sequences to the current chord state ($M_{t-32:t-1} \rightarrow C_{t-32:t}$), leveraging long-term dependencies in musical structure.

### 3. Rhythm/Symphony Generation
Instead of simple block chords, the system generates dynamic, multi-instrument arrangements.
- **Groove Retrieval**: Uses a **K-Nearest Neighbors (k-NN)** approach to match the current melody's rhythm (onset density and pattern) with a massive database of human-composed loops.
- **Tone & Harmony**: The retrieved patterns from the **LPD-5 (Lakh Pianoroll Dataset)** are "snapped" to the predicted chords in real-time.
- **Synthesis**: High-quality audio rendering via **FluidSynth** using curated soundfonts for Drums, Bass, Piano, and Guitar.

---

## 📊 Training & Datasets

### Chord Prediction Models
Our chord estimation models are trained on a diverse corpus of lead sheets and lead-sheet-like data:
- **Pop Dataset**: A collection of 900+ popular songs.
- **EWLD (Encrypted Wikifonia Lead Sheet Dataset)**: Focusing on western popular and folk music.
- **WJazzD (Weimar Jazz Database)**: Providing complex harmonic contexts from jazz solos.

**Pipeline**:
1. **Data Extraction**: Run individual `extract.py` scripts located within `data/*/` (e.g., `data/pop/extract.py`, `data/wjazzd/extract.py`) to clean and convert sources to a unified `.npz` format.
2. **Data Preparation**: `data/prepare_training_data.py` performs data augmentation (transposition, rhythmic shifting) and creates windowed sequences for the Transformer/CRF.

### Rhythm/Symphony Maps
The rhythm/symphony engine is powered by a pre-processed mapping of the **LPD-17 / LPD-5 (Lakh Pianoroll Dataset)**.
- **Extraction**: `data/lpd/extract.py` processes thousands of MIDI files to create a searchable "groove" index in a `.joblib` file.
- **Mapping**: Each melody bar is mapped to its corresponding multi-track accompaniment (Drum, Piano, Guitar, Bass).

---

## 🛠️ Getting Started

### Prerequisites
- Python 3.9+
- [FluidSynth](https://www.fluidsynth.org/) (required for audio synthesis)
- Soundfonts (placed in the `soundfonts/` directory)

### Installation
```bash
pip install -r requirements.txt
```

### Include supporting files
Ensure that you have placed your soundfonts in the `soundfonts/` directory and that the `.joblib` index files are present in `data/lpd/` by running `data/lpd/extract.py`.

### Running the Demo
To start the live transcription and accompaniment system:
```bash
python3 accompaniment/piano_app.py
```

---

## 📈 Model Monitoring
To monitor training progress and visualize loss:
```bash
python3 -m tensorboard.main --logdir=runs --port=6006
```
Visit [http://localhost:6006](http://localhost:6006) to view charts and metrics.
