"""
Per-beat latency benchmark for the harmonic accompaniment pipeline.

Real critical path per beat:
  1. pYIN transcription  (librosa.pyin on one beat of audio, main thread)
  2. chord inference  (AR Transformer OR CRF Viterbi, worker thread)
     -- runs in parallel with --
     k-NN groove query  (independent of chord result, only needs melody groove)
  3. loop sampling  (< 1 ms, negligible)

  End-to-end latency = pYIN + max(chord_inference, k-NN) + loop_sampling

Run from the project root:
    python benchmark_latency.py
"""

import time
import sys
import os
import numpy as np
import librosa

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

N_WARMUP = 20
N_RUNS   = 300

SAMPLE_RATE  = 44100
FRAME_LENGTH = 2048
HOP_LENGTH   = 512
FMIN         = librosa.note_to_hz('C2')
FMAX         = librosa.note_to_hz('C6')


def stats(times_ms):
    a = np.array(times_ms)
    return {
        "mean":  a.mean(),
        "std":   a.std(),
        "p50":   np.percentile(a, 50),
        "p95":   np.percentile(a, 95),
        "p99":   np.percentile(a, 99),
        "min":   a.min(),
        "max":   a.max(),
    }


def print_stats(label, s):
    print(f"\n  {label}")
    print(f"    mean ± std       : {s['mean']:.1f} ± {s['std']:.1f} ms")
    print(f"    p50 / p95 / p99  : {s['p50']:.1f} / {s['p95']:.1f} / {s['p99']:.1f} ms")
    print(f"    min / max        : {s['min']:.1f} / {s['max']:.1f} ms")


# ── 1. pYIN transcription ────────────────────────────────────────────────────

def bench_pyin(bpm=100):
    """librosa.pyin on one beat of synthetic audio at the given BPM."""
    print(f"Benchmarking pYIN transcription @ {bpm} BPM …")
    beat_samples = int(SAMPLE_RATE * 60 / bpm)
    rng = np.random.default_rng(42)

    times = []
    for i in range(N_WARMUP + N_RUNS):
        audio = rng.standard_normal(beat_samples).astype(np.float32) * 0.1
        t0 = time.perf_counter()
        librosa.pyin(audio, fmin=FMIN, fmax=FMAX, sr=SAMPLE_RATE,
                     frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
        t1 = time.perf_counter()
        if i >= N_WARMUP:
            times.append((t1 - t0) * 1000)
    return stats(times)


# ── 2. AR Transformer ────────────────────────────────────────────────────────

def bench_ar_transformer():
    print("Benchmarking AR Transformer …")
    import torch
    from AR.ar_transformer import TransformerModel
    from utils.constants import INPUT_DIM, NUM_CLASSES_ALL, DEVICE, MEMORY

    model = TransformerModel(INPUT_DIM, NUM_CLASSES_ALL).to(DEVICE)
    ckpt = torch.load("checkpoints/transformer_model.pth", map_location=DEVICE)
    model.load_state_dict(ckpt)
    model.eval()

    rng = np.random.default_rng(42)
    times = []
    with torch.no_grad():
        for i in range(N_WARMUP + N_RUNS):
            mel = rng.integers(-1, 89, size=(1, MEMORY, INPUT_DIM)).astype(np.float32)
            x = torch.tensor(mel).to(DEVICE)
            t0 = time.perf_counter()
            out = model(x)
            probs = torch.nn.functional.softmax(out[:, -1, :] / 0.3, dim=-1)
            torch.multinomial(probs, 1)
            t1 = time.perf_counter()
            if i >= N_WARMUP:
                times.append((t1 - t0) * 1000)
    return stats(times)


# ── 3. CRF Viterbi ───────────────────────────────────────────────────────────

def bench_crf():
    print("Benchmarking CRF Viterbi …")
    from CRF.chord_engine import ChordEngine

    engine = ChordEngine()
    rng = np.random.default_rng(42)
    # note[:-1] gives the root, so use e.g. "C4"
    NOTES = ["C4", "D4", "E4", "F4", "G4", "A4", "B4",
             "C#4", "D#4", "F#4", "G#4", "A#4"]

    times = []
    for i in range(N_WARMUP + N_RUNS):
        raw = rng.random(len(NOTES)); raw /= raw.sum()
        bar = dict(zip(NOTES, raw.tolist()))
        engine.reset()
        t0 = time.perf_counter()
        engine.process_bar(bar)
        t1 = time.perf_counter()
        if i >= N_WARMUP:
            times.append((t1 - t0) * 1000)
    return stats(times)


# ── 4. k-NN groove query ─────────────────────────────────────────────────────

def bench_knn():
    print("Benchmarking k-NN groove query …")
    from utils.constants import NN, STEPS_PER_BAR

    rng = np.random.default_rng(42)
    times = []
    for i in range(N_WARMUP + N_RUNS):
        q = rng.random(STEPS_PER_BAR + 1).reshape(1, -1)
        t0 = time.perf_counter()
        NN.kneighbors(q, n_neighbors=10)
        t1 = time.perf_counter()
        if i >= N_WARMUP:
            times.append((t1 - t0) * 1000)
    return stats(times)


# ── 5. Loop sampling ─────────────────────────────────────────────────────────

def bench_loop_sampling():
    print("Benchmarking loop sampling …")
    from utils.constants import NN, DRUMS, PIANOS, GUITARS, BASSES, STEPS_PER_BAR, TEMPERATURE
    from scipy.special import softmax

    rng = np.random.default_rng(42)
    times = []
    for i in range(N_WARMUP + N_RUNS):
        q = rng.random(STEPS_PER_BAR + 1).reshape(1, -1)
        dist, idx = NN.kneighbors(q, n_neighbors=10)
        dist, idx = dist[0], idx[0]
        t0 = time.perf_counter()
        probs = softmax(-dist / TEMPERATURE)
        choice = np.random.choice(len(idx), p=probs)
        _ = (DRUMS[idx[choice]], PIANOS[idx[choice]],
             GUITARS[idx[choice]], BASSES[idx[choice]])
        t1 = time.perf_counter()
        if i >= N_WARMUP:
            times.append((t1 - t0) * 1000)
    return stats(times)


# ── 6. End-to-end: analytical calculation ───────────────────────────────────

def compute_e2e(s_pyin, s_chord, s_knn, s_loop, n=10_000, seed=0):
    """
    Critical path:  pYIN  +  max(chord, k-NN)  +  loop
    All three stages are independent, so we sample from each component's
    empirical distribution (modelled as Gaussian from mean/std) and compute
    the sum analytically over many samples.
    """
    rng = np.random.default_rng(seed)

    def sample(s):
        return np.clip(rng.normal(s["mean"], s["std"], n), s["min"], s["max"])

    pyin  = sample(s_pyin)
    chord = sample(s_chord)
    knn   = sample(s_knn)
    loop  = sample(s_loop)

    parallel = np.maximum(chord, knn)   # the two run in parallel
    total    = pyin + parallel + loop

    return stats(total.tolist())


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Latency benchmark  |  {N_RUNS} timed runs after {N_WARMUP} warm-up runs\n")
    print("=" * 65)

    s_pyin = bench_pyin(bpm=100)
    s_ar   = bench_ar_transformer()
    s_crf  = bench_crf()
    s_knn  = bench_knn()
    s_loop = bench_loop_sampling()

    print("\n" + "=" * 65)
    print("COMPONENT RESULTS\n")
    print_stats("pYIN transcription  @ 100 BPM (one beat = 600 ms audio)", s_pyin)
    print_stats("AR Transformer inference (32-beat context)",               s_ar)
    print_stats("CRF Viterbi inference (one bar)",                          s_crf)
    print_stats("k-NN groove query (k=10, 17-dim)",                        s_knn)
    print_stats("Loop sampling (temperature softmax)",                      s_loop)

    e2e_ar  = compute_e2e(s_pyin, s_ar,  s_knn, s_loop)
    e2e_crf = compute_e2e(s_pyin, s_crf, s_knn, s_loop)

    print("\n" + "=" * 65)
    print("END-TO-END  (pYIN  +  max(chord ∥ k-NN)  +  loop,  100 BPM)\n")
    print_stats("AR path   [pYIN + max(AR, k-NN) + loop]",  e2e_ar)
    print_stats("CRF path  [pYIN + max(CRF, k-NN) + loop]", e2e_crf)

    print("\n" + "=" * 65)
    print("BEAT BUDGET @ 100 BPM  (budget = 600 ms)\n")
    budget = 600
    for label, s in [("AR path", e2e_ar), ("CRF path", e2e_crf)]:
        p95 = s["p95"]
        headroom = budget - p95
        status = "OK" if headroom > 50 else ("MARGINAL" if headroom > 0 else "OVER")
        print(f"  {label:10s}  p95 {p95:.0f} ms  |  headroom {headroom:+.0f} ms  [{status}]")

    # ── BPM vs latency table ─────────────────────────────────────────────────
    # pYIN cost scales linearly with beat duration (proportional to sample count).
    # The parallel stage (Transformer or k-NN) is independent of BPM.
    # e2e(BPM) = pYIN_100 * (100/BPM)  +  max(chord, kNN)
    # headroom(BPM) = (60000/BPM)  -  e2e(BPM)
    pyin_mean_100 = s_pyin["mean"];  pyin_p95_100 = s_pyin["p95"]
    ar_par_mean   = s_ar["mean"];    ar_par_p95   = s_ar["p95"]    # AR dominates k-NN
    crf_par_mean  = s_knn["mean"];   crf_par_p95  = s_knn["p95"]   # k-NN dominates CRF

    print("\n" + "=" * 65)
    print("BPM vs LATENCY  (mean and p95,  headroom = budget - e2e)\n")
    print(f"  {'BPM':>4}  {'Budget':>7}  {'AR mean':>8}  {'AR p95':>7}  {'AR hdroom':>10}  "
          f"{'CRF mean':>9}  {'CRF p95':>8}  {'CRF hdroom':>11}")
    for bpm in [60, 80, 100, 120, 140, 160, 180, 200]:
        budget_ms    = 60_000 / bpm
        scale        = 100 / bpm
        ar_mean      = pyin_mean_100 * scale + ar_par_mean
        ar_p95_      = pyin_p95_100  * scale + ar_par_p95
        crf_mean     = pyin_mean_100 * scale + crf_par_mean
        crf_p95_     = pyin_p95_100  * scale + crf_par_p95
        ar_h  = budget_ms - ar_p95_
        crf_h = budget_ms - crf_p95_
        print(f"  {bpm:>4}  {budget_ms:>6.0f}ms  {ar_mean:>7.0f}ms  {ar_p95_:>6.0f}ms  "
              f"{ar_h:>+9.0f}ms  {crf_mean:>8.0f}ms  {crf_p95_:>7.0f}ms  {crf_h:>+10.0f}ms")
