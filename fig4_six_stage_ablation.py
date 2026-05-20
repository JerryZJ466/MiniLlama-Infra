"""
Fig: Six-Stage Ablation - MatMul Memory Throughput and Decoding Speed
Output: Fig1_Memory_Wall_Ablation.pdf  (replaces old Fig1)
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 8,
    'axes.labelsize': 9,
    'legend.fontsize': 7.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.linewidth': 0.8,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

stages = ['FP32\nBaseline', 'Naive\nINT8', 'Warp-Opt\nINT8',
          '+Fused\nOps', '+Flash\nAttn', '+dp4a\nW8A8']
# S4/S5 inherit S3 memory throughput; S6 is compute-bound (<3 GB/s)
mem_bw     = [58.1, 19.2, 85.6, 85.6, 85.6, 3.2]
throughput = [14.9, 16.1, 31.8, 30.0, 35.2, 44.0]

bar_color  = '#7AA6D1'
line_color = '#C8513C'

fig, ax1 = plt.subplots(figsize=(3.4, 2.4))

bars = ax1.bar(range(6), mem_bw, color=bar_color, width=0.65,
               edgecolor='black', linewidth=0.5, zorder=2,
               label='MatMul mem. throughput (GB/s)')
ax1.set_ylabel('MatMul memory throughput (GB/s)', color='#1A4A7A')
ax1.set_ylim(0, 115)
ax1.set_xticks(range(6))
ax1.set_xticklabels(stages, fontsize=7)
ax1.tick_params(axis='y', colors='#1A4A7A')
ax1.grid(axis='y', linestyle=':', alpha=0.3, zorder=1)
ax1.set_axisbelow(True)

# Label the tiny S6 bar explicitly
ax1.text(5, 5.5, '<3 GB/s\n(compute-bound)', ha='center', va='bottom',
         fontsize=6, color='#1A4A7A',
         bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                   edgecolor='#aaa', alpha=0.85))

ax2 = ax1.twinx()
ax2.plot(range(6), throughput, marker='o', color=line_color,
         markersize=5, linewidth=1.5, zorder=3,
         label='Decoding speed (tok/s)')
for i, v in enumerate(throughput):
    ax2.text(i, v + 1.5, f'{v:.1f}', ha='center', va='bottom',
             fontsize=6.5, color=line_color)
ax2.set_ylabel('Decoding throughput (tok/s)', color=line_color)
ax2.set_ylim(0, 57)
ax2.tick_params(axis='y', colors=line_color)

# Key annotations — minimal and non-overlapping
ax1.annotate('Naive INT8\nslower than FP32',
             xy=(1, 19.2), xytext=(0.0, 68),
             fontsize=6.5, color='#6B2A10', style='italic',
             arrowprops=dict(arrowstyle='->', color='#6B2A10', lw=0.7))

ax2.annotate('+97%', xy=(2, 31.8), xytext=(1.6, 43),
             fontsize=8, color=line_color, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=line_color, lw=0.7))

ax2.annotate('+25%', xy=(5, 44.0), xytext=(4.55, 52),
             fontsize=8, color=line_color, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=line_color, lw=0.7))

# Legend
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc='upper left', frameon=True,
           framealpha=0.95, edgecolor='black', fancybox=False, fontsize=7)

plt.tight_layout()
plt.savefig('Fig1_Memory_Wall_Ablation.pdf', bbox_inches='tight', pad_inches=0.05)
plt.savefig('Fig1_Memory_Wall_Ablation.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
print("Saved: Fig1_Memory_Wall_Ablation.pdf/.png")
