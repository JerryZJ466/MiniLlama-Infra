"""
Fig: Throughput and Energy Efficiency Across Six Stages
Output: Fig5_Energy_Efficiency.pdf  (replaces old Fig5)
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'legend.fontsize': 7.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.linewidth': 0.8,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

stages = ['S1\nFP32', 'S2\nNaive\nINT8', 'S3\nWarp-Opt', 'S4\n+Fused\nOps',
          'S5\n+Flash\nAttn', 'S6\ndp4a\nW8A8']
throughput = [14.9, 16.1, 31.8, 30.0, 35.2, 44.0]
energy = [None, None, None, None, 0.446, 0.584]

bar_color = '#4A7BB7'
bar_hl    = '#D86F45'
line_color = '#B82A2A'

fig, ax1 = plt.subplots(figsize=(3.4, 2.2))

colors = [bar_color] * 5 + [bar_hl]
bars = ax1.bar(range(6), throughput, color=colors, width=0.65,
               edgecolor='black', linewidth=0.5, zorder=2)

for i, v in enumerate(throughput):
    ax1.text(i, v + 0.8, f'{v:.1f}', ha='center', va='bottom',
             fontsize=6.5, fontweight='normal')

ax1.set_ylabel('Decoding throughput (tok/s)')
ax1.set_ylim(0, 52)
ax1.set_xticks(range(6))
ax1.set_xticklabels(stages, fontsize=7)
ax1.grid(axis='y', linestyle=':', alpha=0.3, zorder=1)
ax1.set_axisbelow(True)

ax2 = ax1.twinx()
xs = [4, 5]
ys = [0.446, 0.584]
ax2.plot(xs, ys, marker='D', color=line_color, markersize=5,
         linewidth=1.5, linestyle='--', zorder=3, label='Energy eff. (tok/J)')
ax2.text(4, 0.446 - 0.06, '0.446', ha='center', va='top',
         fontsize=7, color=line_color)
ax2.text(5, 0.584 + 0.03, '0.584', ha='center', va='bottom',
         fontsize=7, color=line_color)
ax2.set_ylabel('Energy efficiency (tok/J)', color=line_color)
ax2.set_ylim(0, 0.75)
ax2.tick_params(axis='y', colors=line_color)

ax2.annotate('+31% tok/J\n(same 76.7 W)',
             xy=(4.5, 0.515), xytext=(2.6, 0.64),
             fontsize=7, color=line_color,
             arrowprops=dict(arrowstyle='->', color=line_color, lw=0.8))

# Power note — bottom right, away from data
ax1.text(0.99, 0.03, 'GPU power: 76.7 W ±4.2 W',
         transform=ax1.transAxes, fontsize=6.5, va='bottom', ha='right',
         bbox=dict(boxstyle='round,pad=0.2', facecolor='#fffff0',
                   edgecolor='#ccc', alpha=0.9))

# Combined legend
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
from matplotlib.patches import Patch
bar_patch = Patch(facecolor=bar_color, edgecolor='black', linewidth=0.5,
                  label='Throughput (tok/s)')
ax1.legend(handles=[bar_patch] + h2, labels=['Throughput (tok/s)'] + l2,
           loc='upper left', frameon=True, framealpha=0.95,
           edgecolor='black', fancybox=False, fontsize=7)

plt.tight_layout()
plt.savefig('Fig5_Energy_Efficiency.pdf', bbox_inches='tight', pad_inches=0.05)
plt.savefig('Fig5_Energy_Efficiency.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
print("Saved: Fig5_Energy_Efficiency.pdf/.png")
