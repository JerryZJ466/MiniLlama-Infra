"""
Fig: Extended Roofline Model on RTX 4060 Laptop
Output: Fig2_Roofline_Model.pdf  (replaces old Fig2)
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 8,
    'axes.labelsize': 9,
    'legend.fontsize': 6.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.linewidth': 0.8,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

fig, ax = plt.subplots(figsize=(3.4, 2.3))

peak_bw        = 272.0    # GB/s
peak_dp4a_gops = 13000.0  # ~13 TOPS dp4a via CUDA cores
peak_tc_gops   = 130000.0 # ~130 TOPS INT8 tensor core (not used by dp4a)

ai = np.logspace(-1, 1.6, 200)

# Memory bandwidth roof
mem_roof = peak_bw * ai   # GOPS

ax.plot(ai, np.minimum(mem_roof, peak_tc_gops),
        '--', color='#555555', linewidth=1.2, label='Mem BW roof (272 GB/s)')
ax.axhline(peak_tc_gops, linestyle='-.', color='#999999', linewidth=0.9,
           label='INT8 tensor core (~130 TOPS)')
ax.axhline(peak_dp4a_gops, linestyle=':', color='#333333', linewidth=1.1,
           label='dp4a CUDA core (~13 TOPS)')

# Shaded memory-bound region
ax.fill_between(ai, 1, np.minimum(mem_roof, peak_tc_gops),
                where=(mem_roof < peak_tc_gops),
                alpha=0.04, color='#4A7BB7')

# Data points: slightly offset S3/S4/S5 so they don't pile up
# (AI, GOPS, label, marker, color)
points = [
    (0.45,  30,   'S1: FP32',        '^', '#888888'),
    (0.65,  20,   'S2: Naive INT8',  'v', '#C07070'),
    (0.85, 155,   'S3: Warp-Opt',    's', '#5BA468'),
    (1.05, 165,   'S4: +Fused Ops',  'D', '#7B9A60'),
    (1.25, 180,   'S5: +Flash Attn', 'P', '#9A8B50'),
    (7.00, 9500,  'S6: dp4a W8A8',   '*', '#C8513C'),
]

for ai_val, perf, label, marker, color in points:
    sz = 120 if marker == '*' else 45
    ax.scatter([ai_val], [perf], marker=marker, s=sz, color=color,
               edgecolor='black', linewidth=0.5, zorder=5, label=label)

ax.annotate('Regime shift', xy=(7.0, 9500), xytext=(2.0, 1800),
            fontsize=7, color='#8B2F1F', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#8B2F1F', lw=0.8))

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(0.3, 15)
ax.set_ylim(10, 300000)
ax.set_xlabel('Arithmetic intensity (Ops/Byte)', labelpad=1)
ax.set_ylabel('Performance (GOPS)')
ax.grid(True, which='both', linestyle=':', alpha=0.25)

# Legend outside, below the axes
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.26),
          frameon=True, framealpha=0.95, edgecolor='black', fancybox=False,
          ncol=3, fontsize=6, handlelength=1.2, handletextpad=0.4,
          columnspacing=0.8, borderpad=0.5)

plt.tight_layout()
plt.subplots_adjust(bottom=0.32)   # reserve space for legend below x-label
plt.savefig('Fig2_Roofline_Model.pdf', bbox_inches='tight', pad_inches=0.05)
plt.savefig('Fig2_Roofline_Model.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
print("Saved: Fig2_Roofline_Model.pdf/.png")
