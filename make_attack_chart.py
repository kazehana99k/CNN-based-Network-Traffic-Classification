"""Generate ATTACK-focused comparison chart from final_comparison.json."""
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PATH = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(PATH, 'outputs', 'final_comparison.json'), encoding='utf-8'))

# Pick 3 configs: paper_orig, paper_improved, and the best (dilated_b64)
configs_to_show = ['paper_orig', 'paper_improved', 'dilated_b64']
labels_pretty = ['Paper (filters 8/16, b128)',
                 'Paper improved (filters 16/32, b64)',
                 'Proposed: Dilated 1D CNN (b64)']

picked = [next(d for d in data if d['config'] == c) for c in configs_to_show]

metrics = ['attack_precision', 'attack_recall', 'attack_f1', 'overall_accuracy']
metric_labels = ['ATTACK Precision', 'ATTACK Recall', 'ATTACK F1', 'Overall Accuracy']

fig, ax = plt.subplots(figsize=(11, 5.5))

x = np.arange(len(metrics))
width = 0.26
colors = ['#9CA3AF', '#3B82F6', '#EF4444']

for i, (cfg, label) in enumerate(zip(picked, labels_pretty)):
    vals = [cfg[m] for m in metrics]
    bars = ax.bar(x + (i - 1) * width, vals, width, label=label, color=colors[i])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f'{v*100:.2f}%',
                ha='center', va='bottom', fontsize=8.5)

ax.set_ylabel('Score')
ax.set_ylim(0.7, 1.02)
ax.set_xticks(x)
ax.set_xticklabels(metric_labels, fontsize=10)
ax.set_title('ATTACK class metrics: proposed method vs original paper variants',
             fontsize=11, pad=12)
ax.legend(loc='lower right', fontsize=9, framealpha=0.95)
ax.grid(True, axis='y', alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()
out = os.path.join(PATH, 'docs', 'attack_comparison.png')
plt.savefig(out, dpi=140, bbox_inches='tight')
print(f'Saved: {out}')
