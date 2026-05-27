"""
Generate chord_snapping_example.png

Real LPD-5 piano loop #1993 (idx=1993) across I-V-vi-IV: C → G → Am → F.
Voice-led 3-voice triads; the number of voices per 16th-note step follows
the groove (4v/3v/1v/rest on beat 2).  Groove voices are mapped directly to
chord tones in rank order so the played notes are always a subset (or slight
extension) of the voiced triad — keeping the figure readable.

Voiced triads (range 48-72, C3-C5):
  C maj  → G maj (G/B: B3-D4-G4, cost 3)  → Am (C4-E4-A4, cost 5) → F (C4-F4-A4, cost 1)
"""

import itertools
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Real loop from LPD-5 (idx 1993) ─────────────────────────────────────────
# counts: [4,4,4,4, 4,3,1,0, 3,3,3,3, 3,3,2,0]
LOOP_COUNTS = [
    [0,5,9,12], [0,5,9,12], [0,5,9,12], [0,5,9,12],   # beat 1  – 4v
    [0,5,9,12], [0,4,7],    [0],         [],            # beat 2  – 4v→3v→1v→0
    [0,4,7],    [0,4,7],    [0,4,7],    [0,4,7],        # beat 3  – 3v
    [0,4,7],    [0,4,7],    [0,3],       [],             # beat 4  – 3v→3v→2v→0
]

# I-V-vi-IV: C maj → G maj → A min → F maj
PROGRESSION = [
    ("C maj",  [0, 4, 7]),   # C E G
    ("G maj",  [7, 11, 2]),  # G B D
    ("A min",  [9, 0, 4]),   # A C E
    ("F maj",  [5, 9, 0]),   # F A C
]

NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

def midi_name(m):
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"

# ── Voice leading ─────────────────────────────────────────────────────────────
def voice_lead(prev_notes, chord_pcs, pitch_range=(48, 72)):
    lo, hi = pitch_range
    candidates = [
        [pc + 12 * o for o in range(-1, 9) if lo <= pc + 12 * o <= hi]
        for pc in chord_pcs
    ]
    if not all(candidates):
        return sorted(pc + 60 for pc in chord_pcs)
    prev_s = sorted(prev_notes[:len(chord_pcs)])
    best, best_cost = None, float("inf")
    for combo in itertools.product(*candidates):
        voicing = sorted(combo)
        if any(voicing[i+1] - voicing[i] < 1 for i in range(len(voicing)-1)):
            continue
        cost = sum(abs(v - p) for v, p in zip(voicing, prev_s))
        if cost < best_cost:
            best_cost, best = cost, voicing
    return best if best else sorted(c[0] for c in candidates)

# ── Compute voiced triads ─────────────────────────────────────────────────────
PITCH_RANGE = (48, 72)   # C3 – C5

voiced_chords = []
prev_voicing = None
for label, pcs in PROGRESSION:
    if prev_voicing is None:
        voicing = voice_lead([60, 64, 67], pcs, PITCH_RANGE)   # seed: close-position C4
    else:
        voicing = voice_lead(prev_voicing, pcs, PITCH_RANGE)
    voiced_chords.append((label, voicing))
    prev_voicing = voicing

print("Voiced chords:")
for (label, v), (_, pcs) in zip(voiced_chords, PROGRESSION):
    prev_idx = [i for i,(l,_) in enumerate(voiced_chords) if l == label][0]
    cost_str = ""
    if prev_idx > 0:
        pv = voiced_chords[prev_idx-1][1]
        c = sum(abs(v[i]-pv[i]) for i in range(len(v)))
        cost_str = f"  cost={c}"
    print(f"  {label}: {[midi_name(n) for n in v]}  {v}{cost_str}")

# ── Map groove voices → chord tones ──────────────────────────────────────────
# n active voices in step → play lowest n chord tones.
# If n > len(triad), double the root one octave up.
def step_notes(n_voices, chord_midi):
    if n_voices == 0:
        return []
    n = len(chord_midi)
    notes = list(chord_midi[:min(n_voices, n)])
    extra = n_voices - n
    if extra > 0:
        root_octave_up = chord_midi[0] + 12
        if root_octave_up <= PITCH_RANGE[1]:
            notes.append(root_octave_up)
    return sorted(set(notes))

played_steps = []
for step_idx, intervals in enumerate(LOOP_COUNTS):
    beat_idx = step_idx // 4
    _, chord_midi = voiced_chords[beat_idx]
    n_voices = len(intervals)
    played_steps.append(step_notes(n_voices, chord_midi))

print("\nPlayed notes per step:")
for i, notes in enumerate(played_steps):
    print(f"  step {i:2d} (beat {i//4+1}): {len(notes)}v  {[midi_name(n) for n in notes]}")

total_disp = sum(
    abs(voiced_chords[i][1][v] - voiced_chords[i-1][1][v])
    for i in range(1, 4)
    for v in range(len(voiced_chords[i][1]))
)
print(f"\nTotal voice displacement: {total_disp} semitones")

# ── Figure ────────────────────────────────────────────────────────────────────
VOICE_COLORS = {
    0: "#e74c3c",   # lowest  – red
    1: "#f39c12",   # middle  – amber
    2: "#2ecc71",   # upper   – green
    3: "#3498db",   # doubled – blue
}

STEP_W, STEP_GAP, NOTE_H = 0.80, 0.20, 0.52

all_midi = [n for step in played_steps for n in step]
y_lo = min(all_midi) - 1
y_hi = max(all_midi) + 1

fig, ax = plt.subplots(figsize=(15, 5.5))
fig.patch.set_facecolor("#1a1a2e")
ax.set_facecolor("#16213e")

# Background grid
for midi in range(y_lo, y_hi + 1):
    pc = midi % 12
    is_black = pc in {1, 3, 6, 8, 10}
    ax.axhline(midi, color="#ffffff05" if not is_black else "#ffffff02",
               linewidth=0.4, zorder=0)

# Beat dividers
for beat in range(5):
    x = beat * 4 * (STEP_W + STEP_GAP) - STEP_GAP / 2
    ax.axvline(x, color="#5a5a8a", linewidth=1.8, zorder=1)
for step in range(1, 16):
    if step % 4 != 0:
        x = step * (STEP_W + STEP_GAP) - STEP_GAP / 2
        ax.axvline(x, color="#252535", linewidth=0.5, zorder=0)

# Note blocks
for step_idx, notes in enumerate(played_steps):
    x = step_idx * (STEP_W + STEP_GAP)
    beat_idx = step_idx // 4
    chord_midi = voiced_chords[beat_idx][1]

    for midi in notes:
        # Colour by rank within voiced chord (0=bass, 1=mid, 2=top, 3=doubled)
        if midi in chord_midi:
            rank = chord_midi.index(midi)
        else:
            rank = 3   # octave-doubled root
        color = VOICE_COLORS[rank]
        rect = mpatches.FancyBboxPatch(
            (x, midi - NOTE_H / 2), STEP_W, NOTE_H,
            boxstyle="round,pad=0.04",
            facecolor=color, edgecolor="white",
            linewidth=0.5, alpha=0.9, zorder=3
        )
        ax.add_patch(rect)
        ax.text(x + STEP_W / 2, midi, midi_name(midi),
                ha="center", va="center",
                fontsize=6.5, color="white", fontweight="bold", zorder=4)

# Voice-leading arrows at beat boundaries
for beat_idx in range(1, 4):
    step_idx = beat_idx * 4
    prev_chord = voiced_chords[beat_idx - 1][1]
    curr_chord = voiced_chords[beat_idx][1]
    for vi in range(len(prev_chord)):
        pm, cm = prev_chord[vi], curr_chord[vi]
        dist = abs(cm - pm)
        color = VOICE_COLORS[vi]
        x_prev = (step_idx - 1) * (STEP_W + STEP_GAP) + STEP_W
        x_curr = step_idx * (STEP_W + STEP_GAP)
        lw = 1.8 if dist <= 2 else 0.9
        alpha = 1.0 if dist <= 2 else 0.6
        ax.annotate(
            "", xy=(x_curr, cm), xytext=(x_prev, pm),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, alpha=alpha),
            zorder=5
        )
        if dist > 0:
            sign = "+" if cm > pm else ""
            mid_x = (x_prev + x_curr) / 2
            mid_y  = (pm + cm) / 2
            offset = 0.4 if cm >= pm else -0.4
            ax.text(mid_x, mid_y + offset, f"{sign}{cm - pm}",
                    ha="center", va="center",
                    fontsize=6.5, color=color, fontweight="bold",
                    alpha=0.95, zorder=6)

# Chord labels + voiced notes
for beat_idx, (label, voicing) in enumerate(voiced_chords):
    x_mid = beat_idx * 4 * (STEP_W + STEP_GAP) + 2 * (STEP_W + STEP_GAP) - STEP_GAP / 2
    ax.text(x_mid, y_hi + 1.05, label,
            ha="center", va="bottom", fontsize=11,
            color="white", fontweight="bold", zorder=4)
    note_str = " – ".join(midi_name(n) for n in voicing)
    ax.text(x_mid, y_hi + 0.4, note_str,
            ha="center", va="bottom", fontsize=7.5,
            color="#bbbbdd", zorder=4)

# Step sub-beat numbers + voice count
for step_idx, notes in enumerate(played_steps):
    x = step_idx * (STEP_W + STEP_GAP) + STEP_W / 2
    sub = step_idx % 4 + 1
    ax.text(x, y_lo - 0.5, str(sub) if notes else "·",
            ha="center", va="top", fontsize=6.5,
            color="#888899" if not notes else "#ccccdd", zorder=4)
    if notes:
        ax.text(x, y_lo - 1.05, f"{len(notes)}v",
                ha="center", va="top", fontsize=5.5,
                color="#555566", zorder=4)

# Total displacement — bottom right, clear of all notes
ax.text(0.995, -0.14,
        f"Total voice displacement: {total_disp} semitones",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=9, color="#2ecc71", fontweight="bold",
        bbox=dict(facecolor="#00000066", edgecolor="none", pad=3))

# Axes
ax.set_xlim(-0.3, 16 * (STEP_W + STEP_GAP) + 0.1)
ax.set_ylim(y_lo - 1.6, y_hi + 2.4)
ax.set_xticks([])
ax.set_yticks(range(y_lo, y_hi + 1))
ax.set_yticklabels(
    [midi_name(m) if m % 12 in {0, 4, 5, 7, 9, 11} else ""
     for m in range(y_lo, y_hi + 1)],
    color="#aaaaaa", fontsize=7.5
)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_color("#444466")
ax.spines["left"].set_color("#444466")
ax.tick_params(colors="#888888")

ax.set_title(
    "Voice-led chord voicing — LPD-5 piano loop #1993\n"
    r"C maj $\to$ G maj $\to$ A min $\to$ F maj  (I – V – vi – IV)",
    color="white", fontsize=11, fontweight="bold", pad=12
)

# Legend — fully below axes
legend_elements = [
    mpatches.Patch(facecolor=VOICE_COLORS[0], label="Bass voice",    edgecolor="white", alpha=0.9),
    mpatches.Patch(facecolor=VOICE_COLORS[1], label="Middle voice",  edgecolor="white", alpha=0.9),
    mpatches.Patch(facecolor=VOICE_COLORS[2], label="Upper voice",   edgecolor="white", alpha=0.9),
    mpatches.Patch(facecolor=VOICE_COLORS[3], label="Doubled root",  edgecolor="white", alpha=0.9),
    mpatches.Patch(facecolor="none", edgecolor="none",
                   label="Arrows: semitone displacement at beat boundary"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=5,
           framealpha=0.25, labelcolor="white", fontsize=8.5,
           facecolor="#16213e", edgecolor="#444466",
           bbox_to_anchor=(0.5, -0.06))

plt.tight_layout(rect=[0, 0.08, 1, 1])

out = "/Users/benlou/Desktop/BenLou/Year 4/FYP/fyp2/report/figures/chord_snapping_example.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\nSaved → {out}")

import shutil
shutil.copy(out, out.replace("report/figures", "public/figures"))
print("Copied → public/figures/chord_snapping_example.png")
