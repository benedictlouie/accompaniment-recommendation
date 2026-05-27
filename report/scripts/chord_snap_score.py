"""
Generate a piano roll showing chord snapping where the chord changes every beat.
Progression (one bar, 4/4): C:maj → F:maj → A:min → G:maj
Same groove as the original example (LPD piano loop 17981).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch

from utils.constants import CHORD_TO_TETRAD
from data.lpd import storage as lpd_storage

# ------------------------------------------------------------------
# Chord snapping (mirrors accompaniment/accompaniment_system.py)
# ------------------------------------------------------------------

def snap_to_chord(target, chord):
    best, best_dist = None, 1e9
    for octave in (-12, 0, 12):
        if octave == 12 and len(chord) > 4:
            continue
        chord_notes = chord if octave != -12 else chord[:4]
        for n in chord_notes:
            candidate = n + octave
            d = abs(candidate - target)
            if d < best_dist:
                best_dist = d
                best = candidate
    return best


def loop_to_midi_notes_per_beat(piano_loop, beat_chords):
    """
    Snap each 16th-note step to the chord active on that beat.
    beat_chords: list of 4 chord names, one per beat (4 steps each).
    """
    result = []
    for step, intervals in enumerate(piano_loop):
        beat = step // 4
        chord_name = beat_chords[beat]
        chord = CHORD_TO_TETRAD[chord_name].copy()
        chord = [n + 12 for n in chord]   # piano raises by one octave
        root = chord[0]

        midi_notes = []
        for interval in intervals:
            if interval < 0:
                continue
            target = root + int(interval)
            midi = snap_to_chord(target, chord)
            if midi not in midi_notes:
                midi_notes.append(midi)
        result.append((step, sorted(midi_notes), chord_name, chord))
    return result


NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def midi_to_name(midi):
    return f"{NOTE_NAMES[midi % 12]}{(midi // 12) - 1}"

# ------------------------------------------------------------------
# Load groove
# ------------------------------------------------------------------

_, PIANOS = lpd_storage.load('piano')
GROOVE = PIANOS[17981]

BEAT_CHORDS = ['C:maj', 'F:maj', 'A:min', 'G:maj']

# Per-beat colour palette: note colour, highlight band, beat bg
BEAT_STYLES = [
    {'note': '#2d6fad', 'band': '#d0e4f5', 'bg': '#eef4fb'},   # C:maj – blue
    {'note': '#5a9e3a', 'band': '#d5edca', 'bg': '#eef7e8'},   # F:maj – green
    {'note': '#b04f25', 'band': '#f5ddd0', 'bg': '#fdf2ec'},   # A:min – terracotta
    {'note': '#7b44b0', 'band': '#e5d5f5', 'bg': '#f5eefb'},   # G:maj – purple
]

snapped = loop_to_midi_notes_per_beat(GROOVE, BEAT_CHORDS)

# Collect all chord tones per beat
beat_chord_tones = {}
for i, name in enumerate(BEAT_CHORDS):
    tones = [n + 12 for n in CHORD_TO_TETRAD[name]]
    beat_chord_tones[i] = tones

# ------------------------------------------------------------------
# Print summary
# ------------------------------------------------------------------

print("Progression: C:maj → F:maj → A:min → G:maj  (one beat each)\n")
for beat, name in enumerate(BEAT_CHORDS):
    tones = beat_chord_tones[beat]
    print(f"  Beat {beat+1}  {name:8s}  tones: {[midi_to_name(n) for n in tones]}")

print()
for step, notes, chord_name, _ in snapped:
    if notes:
        beat = step // 4
        print(f"  step {step:2d} (beat {beat+1}, {chord_name}): {[midi_to_name(n) for n in notes]}")

# ------------------------------------------------------------------
# Piano roll figure — single panel, beat-coloured regions
# ------------------------------------------------------------------

STEPS = 16

all_midi = [n for _, ns, _, _ in snapped for n in ns]
midi_min = min(all_midi) - 1
midi_max = max(all_midi) + 1

fig, ax = plt.subplots(figsize=(14, 7))

# ---- Beat background regions ----
for beat in range(4):
    style = BEAT_STYLES[beat]
    ax.axvspan(beat * 4 - 0.5, beat * 4 + 3.5,
               alpha=0.55, color=style['bg'], zorder=0)

# ---- Chord-tone highlight bands (per beat, only in that beat's x-range) ----
for beat in range(4):
    style = BEAT_STYLES[beat]
    tones = beat_chord_tones[beat]
    x0 = beat * 4 - 0.5
    x1 = beat * 4 + 3.5
    for midi in tones:
        if midi_min <= midi <= midi_max:
            ax.fill_betweenx([midi - 0.48, midi + 0.48], x0, x1,
                             color=style['band'], alpha=0.9, zorder=1)

# ---- Horizontal grid lines ----
for midi in range(midi_min, midi_max + 1):
    ax.axhline(midi, color='#cccccc', linewidth=0.35, zorder=1)

# ---- Note blocks ----
for step, notes, chord_name, chord_tones in snapped:
    beat = step // 4
    style = BEAT_STYLES[beat]
    for midi in notes:
        is_chord_tone = (midi % 12) in {n % 12 for n in chord_tones}
        color = style['note'] if is_chord_tone else '#888888'
        alpha = 1.0 if is_chord_tone else 0.70
        rect = FancyBboxPatch(
            (step - 0.43, midi - 0.4), 0.86, 0.8,
            boxstyle="round,pad=0.02",
            linewidth=0.6,
            edgecolor='white',
            facecolor=color,
            alpha=alpha,
            zorder=3
        )
        ax.add_patch(rect)
        label = NOTE_NAMES[midi % 12]
        ax.text(step, midi, label, ha='center', va='center',
                fontsize=6, color='white', fontweight='bold', zorder=4)

# ---- Beat bar lines ----
for beat in range(1, 4):
    ax.axvline(beat * 4 - 0.5, color='#666666', linewidth=1.0, zorder=2)

# ---- Beat chord label at top ----
for beat, name in enumerate(BEAT_CHORDS):
    root, q = name.split(':')
    quality = 'maj' if q == 'maj' else 'min'
    tones = beat_chord_tones[beat]
    tones_str = ' – '.join(NOTE_NAMES[n % 12] for n in tones)
    style = BEAT_STYLES[beat]
    ax.text(beat * 4 + 1.5, midi_max + 0.8,
            f"{root} {quality}\n({tones_str})",
            ha='center', va='bottom', fontsize=8, fontweight='bold',
            color=style['note'])

# ---- Y-axis ----
yticks = list(range(midi_min, midi_max + 1))
ylabels = []
for m in yticks:
    name = NOTE_NAMES[m % 12]
    octave = (m // 12) - 1
    ylabels.append(f"C{octave}" if name == 'C' else name)
ax.set_yticks(yticks)
ax.set_yticklabels(ylabels, fontsize=7)
ax.set_ylim(midi_min - 0.6, midi_max + 1.4)
ax.set_xlim(-0.6, STEPS - 0.4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ---- X-axis ----
ax.set_xticks([i * 4 for i in range(4)])
ax.set_xticklabels([f'Beat {i+1}' for i in range(4)], fontsize=9)
ax.set_xticks(range(STEPS), minor=True)
ax.tick_params(axis='x', which='minor', length=3, color='#bbbbbb')

# ---- Legend ----
from matplotlib.lines import Line2D

legend_elements = [
    Patch(facecolor=BEAT_STYLES[i]['note'],
          label=f"Beat {i+1} — {BEAT_CHORDS[i].replace(':', ' ')}  "
                f"({' – '.join(NOTE_NAMES[n % 12] for n in beat_chord_tones[i])})")
    for i in range(4)
]

ax.legend(handles=legend_elements, loc='lower right', fontsize=7.5,
          framealpha=0.95, edgecolor='#bbbbbb',
          title='Beat chord  /  note colour', title_fontsize=8)

# ---- Titles ----
fig.suptitle(
    'Chord Snapping with a Changing Chord Progression\n'
    'C maj → F maj → A min → G maj   (LPD piano loop 17981)',
    fontsize=11, fontweight='bold', y=1.01
)
ax.set_xlabel('16th-note steps (one bar, 4/4)', fontsize=9)

plt.tight_layout()

out_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'figures',
                 'chord_snapping_example.png')
)
# also overwrite if previously saved under old name
plt.savefig(out_path, dpi=180, bbox_inches='tight', facecolor='white')
print(f"\nSaved to {out_path}")
