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

const ROOTS                = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];
const BASE_OCTAVE          = 36;   // C2 — matches Python CHORD_TO_TETRAD
const API_BASE             = '/api';
const BEATS_PER_BAR        = 4;
const STEPS_PER_BEAT       = 4;    // MELODY_NOTES_PER_BEAT
const LATENCY_COMPENSATION = 3;    // 16th-note steps to fire AR prediction early (0.75 of a beat)

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

// Active engine: "transformer" | "crf"
let selectedEngine = 'transformer';

// CRF state round-tripped as base64 (stateless API pattern)
let crfDelta        = '';
let crfBarHistory   = '';
let crfBarPitch     = '';
let crfBeatCount    = 0;
let crfLoopHistory  = '';
// Chord predicted at bar-end (beat 4), applied at next bar-start (beat 1)
let crfPendingChord = '';

/** Reset all CRF round-trip state. */
function resetCRFState() {
  crfDelta = ''; crfBarHistory = ''; crfBarPitch = ''; crfBeatCount = 0;
  crfLoopHistory = ''; crfPendingChord = '';
}

// Shared API / accompaniment state
let engineHistory  = '';    // base64 history round-tripped with the transformer engine
let prevDrumLoop   = '';    // base64 drum loop round-tripped for quality fallback
let pendingArChord = null;  // chord pre-computed by AR early-fire
let earlyApiCallId = 0;     // prevents stale early-fire responses
let predictedChord = '–';   // last chord returned by the prediction API
let pendingLoops   = null;  // next bar's loops (set by API response, promoted at beat 1)

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
 * Depends on page-level global: audioContext.
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
// SHARED API HELPERS
// =============================================================

/**
 * Fire AR prediction early (LATENCY_COMPENSATION 16th-note steps before the beat).
 * Sends empty notes so the server peeks using history alone; stores the result in
 * pendingArChord so the beat handler can apply it synchronously when the beat fires.
 */
async function predictChordEarlyAR(beat, beatTime, beatDur, beatFireMs) {
  const callId = ++earlyApiCallId;
  try {
    const fetchStart = performance.now();
    const res = await fetch(`${API_BASE}/predict-chord`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        notes: [],
        beat_start:           beatTime - beatDur,
        beat_index:           beat,
        tempo,
        engine:               'transformer',
        history:              engineHistory,
        prev_drum_loop:       prevDrumLoop,
        early_fire:           true,
        latency_compensation: LATENCY_COMPENSATION,
      }),
    });
    const data = await res.json();
    const serverMs = (performance.now() - fetchStart).toFixed(0);
    const totalMs  = (performance.now() - beatFireMs).toFixed(0);

    if (callId !== earlyApiCallId) return;  // superseded by a newer early-fire call

    if (data.chord) {
      pendingArChord = data.chord;
      // Do NOT update predictedChord or the display here — both change at beat
      // boundaries (onBeat / harmoniser beat-check) so audio and UI stay in sync.
      console.log(`[CHORD] beat=${beat}  chord=${data.chord}  total=${totalMs}ms  server=${serverMs}ms`);
    }
  } catch (e) {}
}

/** Apply CRF round-trip state returned by the predict-chord API response. */
function updateCRFFromResponse(data) {
  crfDelta       = data.crf_delta        ?? crfDelta;
  crfBarHistory  = data.crf_bar_history  ?? crfBarHistory;
  crfBarPitch    = data.crf_bar_pitch    ?? crfBarPitch;
  crfBeatCount   = data.crf_beat_count   ?? crfBeatCount;
  crfLoopHistory = data.crf_loop_history ?? crfLoopHistory;
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

/**
 * Page-specific stop callback — set by each page so the engine
 * selector in shared.js can stop+reset when the user switches engines.
 */
let onEngineSwitch = () => {};

document.querySelectorAll('.engine-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    if (btn.dataset.engine === selectedEngine) return;
    selectedEngine = btn.dataset.engine;
    document.querySelectorAll('.engine-btn').forEach(b =>
      b.classList.toggle('active', b.dataset.engine === selectedEngine));
    resetCRFState();
    onEngineSwitch();
  });
});

// =============================================================
// SOUNDFONT  (shared by piano.html and harmoniser.html)
// Each page defines SOUNDFONT_INSTRUMENTS = { piano, guitar, bass }.
// =============================================================

const sf    = { piano: null, guitar: null, bass: null };
let sfReady = 0;

function loadSoundfonts(instruments) {
  if (typeof Soundfont === 'undefined') return;
  Object.entries(instruments).forEach(([key, name]) => {
    Soundfont.instrument(audioContext, name, { soundfont: 'MusyngKite' })
      .then(player => { sf[key] = player; sfReady++; updateSfStatus(); })
      .catch(err => console.warn('Soundfont load failed:', key, err));
  });
}

function updateSfStatus() {
  const el = document.getElementById('api-status');
  el.textContent = sfReady < 3
    ? `connected · loading sounds (${sfReady}/3)…`
    : 'connected · sounds ready';
  el.className = 'status-ok';
}
