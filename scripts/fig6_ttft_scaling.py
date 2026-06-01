"""
Fig: TTFT Scaling with Context Length (Stage 5, RTX 4060)
Output: Fig3_Context_Scaling.pdf  (replaces old Fig3)
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

ctx_len  = [53, 85, 149, 277, 533]
ttft_s   = [t / 1000 for t in [1604.5, 2521.9, 4427.5, 8188.4, 15792.7]]

# O(N) ideal reference anchored at first point
ideal_on = [ttft_s[0] * (c / ctx_len[0]) for c in ctx_len]

fig, ax = plt.subplots(figsize=(3.4, 2.3))

ax.plot(ctx_len, ttft_s, marker='o', color='#C8513C', linewidth=1.8,
        markersize=6, label='Measured (Stage 5)', zorder=3)
ax.plot(ctx_len, ideal_on, '--', color='#888888', linewidth=1.2,
        label='Ideal O(N) reference', zorder=2)

ax.fill_between(ctx_len, ideal_on, ttft_s, alpha=0.12,
                color='#C8513C', zorder=1)

# Annotate data points
for c, t in zip(ctx_len, ttft_s):
    ax.text(c, t + 0.35, f'{t:.1f}s', ha='center', va='bottom',
            fontsize=6.5, color='#C8513C')

ax.annotate(r'$O(N^2)$ prefill bottleneck',
            xy=(390, 11.5), xytext=(120, 14.5),
            fontsize=7.5, color='#6B2A10', style='italic',
            arrowprops=dict(arrowstyle='->', color='#6B2A10', lw=0.8))

ax.set_xlabel('Context length (tokens)')
ax.set_ylabel('Time to first token (s)')
ax.set_xlim(0, 600)
ax.set_ylim(0, 18)
ax.set_xticks(ctx_len)
ax.grid(True, linestyle=':', alpha=0.3)
ax.legend(loc='upper left', frameon=True, framealpha=0.95,
          edgecolor='black', fancybox=False)

plt.tight_layout()
plt.savefig('Fig3_Context_Scaling.pdf', bbox_inches='tight', pad_inches=0.05)
plt.savefig('Fig3_Context_Scaling.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
print("Saved: Fig3_Context_Scaling.pdf/.png")
