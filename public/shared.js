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
