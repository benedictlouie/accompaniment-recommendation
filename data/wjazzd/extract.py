import sqlite3
import numpy as np
import os
import re

# ------------------
# Settings
# ------------------

DB_PATH = "data/wjazzd/wjazzd.db"
OUTPUT_DIR = "data/wjazzd/quantised_npz"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------
# Load data
# ------------------

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Beats (exclude pickup bars)
cursor.execute("""
SELECT melid, bar, beat, onset, chord
FROM beats
WHERE bar >= 1
ORDER BY melid, onset
""")
beats_all = cursor.fetchall()

# Melody (exclude pickup bars)
cursor.execute("""
SELECT melid, bar, onset, duration, pitch
FROM melody
WHERE pitch IS NOT NULL AND bar >= 1
ORDER BY melid, onset
""")
melody_all = cursor.fetchall()

conn.close()

# ------------------
# Organize by solo
# ------------------

beats_by_mel = {}
for melid, bar, beat, onset, chord in beats_all:
    beats_by_mel.setdefault(melid, []).append(
        dict(bar=bar, beat=beat, onset=onset, chord=chord)
    )

melody_by_mel = {}
for melid, bar, onset, duration, pitch in melody_all:
    melody_by_mel.setdefault(melid, []).append(
        dict(onset=onset, duration=duration, pitch=int(round(pitch)))
    )

all_melids = sorted(set(beats_by_mel.keys()))
print(f"Found {len(all_melids)} solos.")

# Chord conversion
def convert(chord):
    # Extract root (only first note part)
    chord = chord.split('/')[0]
    match = re.match(r'^([A-G])([#b]?)(.*)', chord)
    note = match.group(1)
    accidental = match.group(2)
    rest = match.group(3)
    root = note + accidental

    # Quality mapping (ignore +/- here)
    rest = rest.replace('+', 'aug').replace('-', 'min')
    quality = rest.split(' ')[0]
    quality = re.sub(r'^m(\d*)$', r'min\1', quality)
    quality = re.sub(r'^M(\d*)$', r'maj\1', quality)
    quality = re.sub(r'^j(\d*)$', r'maj\1', quality)
    quality = quality.replace('minj', 'mM')
    
    if 'o7' in rest:
        quality = 'dim7'
    elif 'o' in rest:
        quality = 'dim'
    elif 'ø' in rest:
        quality = 'm7b5'
    elif 'sus4' in rest:
        quality = 'sus4'
    elif 'sus' in rest:
        quality = 'sus2'
    elif quality == '':
        quality = 'maj'
    return f"{root}:{quality}"


# ------------------
# Process each solo
# ------------------

for melid in all_melids:

    beats = beats_by_mel.get(melid, [])
    melody = melody_by_mel.get(melid, [])

    if not beats:
        continue

    # --------------------------------
    # Build beat info
    # --------------------------------
    beat_onsets = np.array([b["onset"] for b in beats])
    beat_numbers = np.array([b["beat"] for b in beats])

    # Median beat duration
    beat_durations = np.diff(beat_onsets)
    if len(beat_durations) == 0:
        continue
    median_beat_duration = np.median(beat_durations)

    # 16th-note duration
    sixteenth_duration = median_beat_duration / 4.0

    # Piece timing
    start_time = beat_onsets[0]
    end_time = beat_onsets[-1] + median_beat_duration

    total_beats = len(beats)
    total_16ths = int(np.ceil((end_time - start_time) / sixteenth_duration))

    # --------------------------------
    # Initialize arrays
    # --------------------------------
    strong_beats = np.zeros(total_beats, dtype=bool)
    chords = np.array([""] * total_beats, dtype="<U32")
    midi_notes = np.full(total_16ths, -1, dtype=int)

    # --------------------------------
    # Fill beats + strong beats + chords
    # --------------------------------
    for i, b in enumerate(beats):
        strong_beats[i] = (b["beat"] == 1)
        chord = b["chord"]
        if not chord or 'N' in chord:
            chord = "N"
        else:
            chord = convert(chord)
        chords[i] = chord if chord not in (None, "") else ""

    # Forward-fill chords
    last = None
    for i in range(total_beats):
        if chords[i] == "":
            if last is not None:
                chords[i] = last
            else:
                chords[i] = "N"
        else:
            last = chords[i]

    # --------------------------------
    # Fill melody aligned to beats
    # --------------------------------
    for note in melody:

        onset = note["onset"]
        duration = note["duration"]
        pitch = note["pitch"]

        # Find nearest beat on the left
        beat_idx = np.searchsorted(beat_onsets, onset) - 1
        if beat_idx < 0 or beat_idx >= len(beats):
            continue

        beat_start = beat_onsets[beat_idx]

        # Fractional position inside the beat
        frac = (onset - beat_start) / median_beat_duration
        offset_16 = int(round(frac * 4))

        start_idx = beat_idx * 4 + offset_16
        dur_16 = int(round(duration / sixteenth_duration))
        end_idx = start_idx + dur_16

        start_idx = max(0, start_idx)
        end_idx = min(total_16ths, end_idx)

        midi_notes[start_idx:end_idx] = pitch

    # --------------------------------
    # Save NPZ
    # --------------------------------
    filename = f"solo_{melid:03d}.npz"
    out_path = os.path.join(OUTPUT_DIR, filename)

    np.savez(
        out_path,
        strong_beats=strong_beats,
        melody=midi_notes,
        chords=chords
    )

    print(f"Saved {filename} | beats={total_beats} | 16ths={total_16ths}")

print("Done.")