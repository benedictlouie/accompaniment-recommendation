import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

B = np.linspace(55, 210, 400)

bpms = [60, 80, 100, 120, 140, 160, 180, 200]
ar_p95  = [420, 345, 299, 269, 247, 231, 219, 209]
crf_p95 = [332, 256, 211, 180, 159, 143, 130, 120]

fig, ax = plt.subplots(figsize=(7, 4))

ax.fill_between(B, 50,           15000/B, color='steelblue',  alpha=0.10)
ax.fill_between(B, 15000/B, 30000/B, color='teal',       alpha=0.13)
ax.fill_between(B, 30000/B, 45000/B, color='darkorange', alpha=0.13)

for n, label in [(1, r'$1\times\delta_{16}$'),
                 (2, r'$2\times\delta_{16}$'),
                 (3, r'$3\times\delta_{16}$')]:
    ax.plot(B, n * 15000 / B, color='grey', lw=0.8, ls='--')
    ax.text(207, n * 15000 / 207 + 4, label, fontsize=7, color='grey',
            va='bottom', ha='right')

ax.text(195, 65,  'ext. = 1', fontsize=7, color='grey', ha='right')
ax.text(195, 155, 'ext. = 2', fontsize=7, color='grey', ha='right')
ax.text(195, 275, 'ext. = 3', fontsize=7, color='grey', ha='right')

ax.plot(bpms, ar_p95,  'o-', color='steelblue', lw=1.8, ms=5, label='AR p95')
ax.plot(bpms, crf_p95, 's-', color='firebrick', lw=1.8, ms=5, label='CRF p95')

ax.set_xlabel('Tempo (BPM)', fontsize=10)
ax.set_ylabel('p95 latency (ms)', fontsize=10)
ax.set_xlim(55, 210)
ax.set_ylim(50, 480)
ax.set_xticks(bpms)
ax.grid(True, color='grey', alpha=0.2)
ax.legend(fontsize=9, loc='upper right')
fig.tight_layout()
fig.savefig('../figures/latency_extension.png', dpi=200, bbox_inches='tight')
