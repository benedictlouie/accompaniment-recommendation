/**
 * shared.js — constants, state, and utilities used by both
 *             piano.html and harmoniser.html.
 *
 * Load this script BEFORE the page-specific inline <script>.
 * Because all scripts sit at the bottom of <body>, the DOM is
 * already fully parsed when this runs.
 */

// =============================================================
// CONSTANTS  (mirrors utils/constants.py)
// =============================================================

const ROOTS          = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];
const BASE_OCTAVE    = 36;   // C2 — matches Python CHORD_TO_TETRAD
const API_BASE       = '/api';
const BEATS_PER_BAR  = 4;
const STEPS_PER_BEAT = 4;    // MELODY_NOTES_PER_BEAT

// Interval offsets per quality (mirrors Python CHORD_TO_TETRAD / QUALITIES_ALL).
// Used by piano.html (chordMidis) and harmoniser.html (CHORD_TETRAD).
const CHORD_INTERVALS = {
  maj:  [0,4,7,12], min:  [0,3,7,12],
  maj7: [0,4,7,11], min7: [0,3,7,10],
  aug:  [0,4,8,12], dim:  [0,3,6,12], dim7: [0,3,6,9],
  sus2: [0,2,7,12], sus4: [0,5,7,12],
  '7':  [0,4,7,10], '6':  [0,4,7,9],  min6: [0,3,7,9],
  m7b5: [0,3,6,10], mM7:  [0,3,7,11],
};

// Absolute-MIDI tetrad per chord label (root pinned to BASE_OCTAVE = C2).
// harmoniser.html uses this for voice-leading; piano.html uses chordMidis() instead.
const CHORD_TETRAD = { N: [-1,-1,-1,-1] };
ROOTS.forEach((r, i) => {
  const root = BASE_OCTAVE + i;
  for (const [q, ivs] of Object.entries(CHORD_INTERVALS)) {
    CHORD_TETRAD[`${r}:${q}`] = ivs.map(iv => root + iv);
  }
});

// =============================================================
// SHARED MUTABLE STATE
// =============================================================

// Current tempo in BPM — modified by setTempo(); read by page scripts.
let tempo = 100;

// Per-instrument on/off flags — toggled by the UI buttons below.
const instrumentOn = { piano: true, guitar: true, bass: true, drums: true, metronome: true };

// =============================================================
// SHARED FUNCTIONS
// =============================================================

/**
 * Convert a note name (e.g. "A#4") to a MIDI integer.
 * Used by both piano.html (key mapping) and harmoniser.html (pitch visualiser).
 */
function noteNameToMidi(name) {
  const root = name.slice(0, -1);
  const oct  = parseInt(name.slice(-1));
  const idx  = ROOTS.indexOf(root);
  return idx >= 0 ? (oct + 1) * 12 + idx : -1;
}

/**
 * Play one note via soundfont at absolute AudioContext time t.
 * Passes a raw MIDI integer so the library resolves the correct
 * flat-notation sample key (e.g. Bb4, not A#4).
 * Depends on page-level globals: sf (soundfont players), audioContext.
 */
function sfPlay(inst, midi, t, dur, gain) {
  const player = sf[inst];
  if (!player) return;
  const playAt = Math.max(audioContext.currentTime + 0.003, t);
  player.play(midi, playAt, { duration: dur, gain: gain });
}

/** Clamp BPM to [20, 300] and update the display. */
function setTempo(bpm) {
  tempo = Math.max(20, Math.min(300, bpm));
  document.getElementById('bpm-display').textContent = `${tempo} BPM`;
}

/** Highlight the active beat dot (1-indexed). */
function updateBeatDots(beat) {
  document.querySelectorAll('.beat-dot').forEach(dot => {
    dot.classList.toggle('active', parseInt(dot.dataset.beat) === beat);
  });
}

// =============================================================
// DOM INIT — tempo buttons + instrument toggles
// (DOM is ready because this script is at the end of <body>)
// =============================================================

document.getElementById('btn-plus' ).addEventListener('click', () => setTempo(tempo + 5));
document.getElementById('btn-minus').addEventListener('click', () => setTempo(tempo - 5));

document.querySelectorAll('.toggle-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const inst = btn.dataset.inst;
    instrumentOn[inst] = !instrumentOn[inst];
    btn.classList.toggle('on',  instrumentOn[inst]);
    btn.classList.toggle('off', !instrumentOn[inst]);
  });
});
