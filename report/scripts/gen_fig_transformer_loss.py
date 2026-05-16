"""
Plot AR Transformer train + validation loss from a specific TensorBoard event file.
Run from the repo root:
    python report/scripts/gen_fig_transformer_loss.py

Uses: runs/transformer_model/events.out.tfevents.1773489543.kalman.2441875.0
  Loss/train      – CE loss logged every ~142 steps (10 000 pts total)
  Loss/validation – CE loss logged every ~9 500 steps (150 pts total)

Outputs: report/figures/transformer_loss.png
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

EVENT_FILE = "runs/transformer_model/events.out.tfevents.1773489543.kalman.2441875.0"
OUT_PATH   = "report/figures/transformer_loss.png"
STRIDE     = 5   # plot every 5th train point (10 000 → 2 000 pts)

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

if not os.path.exists(EVENT_FILE):
    print(f"Event file not found: {EVENT_FILE}")
    sys.exit(1)

ea = EventAccumulator(EVENT_FILE)
ea.Reload()
tags = ea.Tags().get('scalars', [])
print("Available tags:", tags)

train_scalars = ea.Scalars('Loss/train')
val_scalars   = ea.Scalars('Loss/validation')

train_steps = np.array([s.step  for s in train_scalars])
train_vals  = np.array([s.value for s in train_scalars])
val_steps   = np.array([s.step  for s in val_scalars])
val_vals    = np.array([s.value for s in val_scalars])

print(f"Train: {len(train_steps)} pts, steps {train_steps[0]}–{train_steps[-1]}, "
      f"first={train_vals[0]:.4f}, last={train_vals[-1]:.4f}")
print(f"Test:  {len(val_steps)} pts, steps {val_steps[0]}–{val_steps[-1]}, "
      f"first={val_vals[0]:.4f}, last={val_vals[-1]:.4f}")

fig, ax = plt.subplots(figsize=(6, 3.5))

s = max(1, STRIDE)
ax.plot(train_steps[::s], train_vals[::s],
        color='steelblue', linewidth=0.9, alpha=0.85, label='Train loss')
ax.plot(val_steps, val_vals,
        color='firebrick', linewidth=1.5, linestyle='--', label='Test loss')

ax.set_xlabel("Training step")
ax.set_ylabel("Cross-Entropy Loss")
ax.legend(frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
print(f"Saved → {OUT_PATH}")
