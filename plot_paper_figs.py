import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 8.5

# ─────────────────────────────────────────────
# Fig 1: Six-Stage Ablation (throughput + matmul BW)
# ─────────────────────────────────────────────
def plot_memory_wall_ablation():
    fig, ax1 = plt.subplots(figsize=(8, 4))

    labels = ['FP32\nBaseline', 'Naive\nINT8', 'Warp-Opt\nINT8',
              '+Fused\nOps', '+Flash\nAttn', '+dp4a\nW8A8']
    # MatMul memory throughput (GB/s); Stage 6 is compute-bound — report as effective ~3 GB/s
    mem_throughput = [58.1, 19.2, 85.6, 85.6, 85.6, 3.2]
    decoding_speed  = [14.9, 16.1, 31.8, 30.0, 35.2, 44.0]
    bar_colors = ['#888888', '#F5A623', '#4A90E2', '#7ED321', '#9013FE', '#D0021B']

    x = np.arange(len(labels))
    width = 0.45

    bars = ax1.bar(x, mem_throughput, width, color=bar_colors, alpha=0.82,
                   label='MatMul Memory Throughput (GB/s)', zorder=2)
    ax1.set_ylabel('MatMul Memory Throughput (GB/s)', fontweight='bold')
    ax1.set_ylim(0, 115)
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()
    ax2.plot(x, decoding_speed, color='#D0021B', marker='o', linewidth=2.5,
             markersize=8, label='Decoding Throughput (tok/s)', zorder=5)
    ax2.set_ylabel('Decoding Throughput (tok/s)', color='#D0021B', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#D0021B')
    ax2.set_ylim(0, 55)

    # Annotations — carefully positioned to avoid overlap
    ax2.annotate('Naive INT8\nslower than FP32!', xy=(1, 16.1), xytext=(0.1, 6),
                 arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                 fontsize=8, color='red', fontstyle='italic')
    ax2.annotate('+97%', xy=(2, 31.8), xytext=(2.6, 38),
                 arrowprops=dict(arrowstyle='->', color='#1a7a1a', lw=1.5),
                 fontsize=9.5, color='#1a7a1a', fontweight='bold')
    ax2.annotate('+25%', xy=(5, 44.0), xytext=(4.55, 51),
                 arrowprops=dict(arrowstyle='->', color='#900', lw=1.5),
                 fontsize=9, color='#900', fontweight='bold')
    # Stage 6: label the tiny bar explaining compute-bound
    ax1.text(5, 6, 'compute-\nbound\n(<3 GB/s)', ha='center', va='bottom',
             fontsize=7, color='gray',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='gray'))

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper left', framealpha=0.9, fontsize=8.5)

    plt.title('Six-Stage Ablation: Memory Throughput and Decoding Speed (RTX 4060 Laptop)',
              fontsize=10, pad=8)
    plt.tight_layout()
    plt.savefig('Fig1_Memory_Wall_Ablation.pdf', format='pdf', dpi=300, bbox_inches='tight')
    print("Done: Fig1_Memory_Wall_Ablation.pdf")


# ─────────────────────────────────────────────
# Fig 2: Roofline Model (extended to Stage 6)
# ─────────────────────────────────────────────
def plot_roofline_model():
    fig, ax = plt.subplots(figsize=(6, 4.2))

    mem_bw   = 272.0   # RTX 4060 peak GB/s
    peak_int8_tops = 130.0  # RTX 4060 INT8 TOPS (approximate)

    # Roof lines
    x_line = np.logspace(-1, 2, 300)
    y_bw = mem_bw * x_line

    dp4a_cuda_core_tops = 13.0  # dp4a via CUDA cores, not tensor cores

    ax.plot(x_line, y_bw, color='black', linestyle='--', linewidth=1.5,
            label=f'Mem BW roof ({mem_bw} GB/s)')
    ax.plot(x_line, [peak_int8_tops*1000]*len(x_line), color='navy',
            linestyle=':', linewidth=1.0, alpha=0.5,
            label=f'INT8 Tensor Core roof (~{peak_int8_tops} TOPS)')
    ax.plot(x_line, [dp4a_cuda_core_tops*1000]*len(x_line), color='#D0021B',
            linestyle='-.', linewidth=1.4,
            label=f'dp4a CUDA Core roof (~{dp4a_cuda_core_tops} TOPS)')

    # Data points: (arithmetic_intensity, performance_GOPS, label, color, marker)
    points = [
        (0.50, 29.0,   'S1: FP32 Baseline',  '#888888', 's'),
        (1.00, 19.0,   'S2: Naive INT8',      '#F5A623', 'v'),
        (1.00, 160.0,  'S3: Warp-Opt INT8',   '#4A90E2', '^'),
        (1.00, 160.0,  'S4: +Fused Ops',      '#7ED321', 'D'),
        (1.00, 188.0,  'S5: +Flash Attn',     '#9013FE', 'p'),
        (4.00, 350.0,  'S6: dp4a W8A8',       '#D0021B', '*'),
    ]

    for intensity, perf, label, color, marker in points:
        ax.scatter(intensity, perf, color=color, marker=marker, s=130,
                   label=label, zorder=6, edgecolors='black', linewidths=0.5)

    # Annotate regime transition
    ax.axvline(x=mem_bw/peak_int8_tops/1000 * 10, color='gray', linestyle=':', alpha=0.4)
    ax.text(1.8, 25, 'Memory-\nbound', fontsize=8, color='gray', ha='center')
    ax.text(5.5, 25, 'Compute-\nbound', fontsize=8, color='navy', ha='center')
    ax.annotate('Regime\nshift', xy=(4.0, 350), xytext=(6, 120),
                arrowprops=dict(arrowstyle='->', color='#D0021B', lw=1.5),
                fontsize=8, color='#D0021B', fontweight='bold')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.2, 15)
    ax.set_ylim(10, 2000)
    ax.set_xlabel('Arithmetic Intensity (FLOPs/Byte)', fontweight='bold')
    ax.set_ylabel('Performance (GOPS)', fontweight='bold')
    ax.grid(True, which='both', ls='--', alpha=0.25)
    ax.legend(loc='lower right', framealpha=0.9, ncol=2, fontsize=7.5)
    plt.title('Roofline Analysis: RTX 4060 Laptop', fontsize=10, pad=6)
    plt.tight_layout()
    plt.savefig('Fig2_Roofline_Model.pdf', format='pdf', dpi=300, bbox_inches='tight')
    print("Done: Fig2_Roofline_Model.pdf")


# ─────────────────────────────────────────────
# Fig 3: Context Scaling TTFT
# ─────────────────────────────────────────────
def plot_context_scaling():
    fig, ax = plt.subplots(figsize=(6, 4))

    context_lengths = [53, 85, 149, 277, 533]
    ttft_seconds    = [1.604, 2.521, 4.427, 8.188, 15.792]

    ax.plot(context_lengths, ttft_seconds, marker='D', color='#9013FE',
            linewidth=2.5, markersize=8, label='MiniLlama-Infra Stage 5')
    ax.fill_between(context_lengths, ttft_seconds, color='#9013FE', alpha=0.08)

    scale = ttft_seconds[0] / context_lengths[0]
    ideal = [l * scale for l in context_lengths]
    ax.plot(context_lengths, ideal, '--', color='gray', linewidth=1,
            alpha=0.7, label='Ideal O(N) reference')

    ax.set_xlabel('Context Length (Tokens)', fontweight='bold')
    ax.set_ylabel('Time to First Token (Seconds)', fontweight='bold')
    ax.set_xticks(context_lengths)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.annotate('O(N²) prefill\nbottleneck',
                xy=(277, 8.188), xytext=(100, 11),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6),
                fontsize=10)
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('Fig3_Context_Scaling.pdf', format='pdf', dpi=300, bbox_inches='tight')
    print("Done: Fig3_Context_Scaling.pdf")


# ─────────────────────────────────────────────
# Fig 4 (NEW): Memory Access Pattern Diagram
# ─────────────────────────────────────────────
def plot_memory_access_diagram():
    fig, (ax_naive, ax_warp) = plt.subplots(1, 2, figsize=(9, 4.5))
    fig.patch.set_facecolor('#FAFAFA')

    # ─── Left: Naive INT8 ───
    ax = ax_naive
    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(-0.3, 4.0)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Stage 2: Naive INT8\n(Uncoalesced Access)', fontsize=11,
                 fontweight='bold', color='#C0392B', pad=8)

    # Memory row (stride = 2048 bytes apart conceptually, drawn compressed)
    mem_xs = [i * 0.95 for i in range(8)]
    for i, mx in enumerate(mem_xs):
        rect = Rectangle((mx, 0.1), 0.8, 0.45, linewidth=1.2,
                          edgecolor='#2C3E50', facecolor='#FAD7A0', zorder=2)
        ax.add_patch(rect)
        ax.text(mx + 0.4, 0.32, f'Row {i}', ha='center', va='center', fontsize=6.5)

    ax.text(3.5, -0.2, '← stride = 2048 bytes →', ha='center', fontsize=8,
            color='#C0392B', style='italic')

    # Threads
    thread_xs = [i * 0.95 for i in range(8)]
    for i, tx in enumerate(thread_xs):
        rect = Rectangle((tx, 2.6), 0.8, 0.45, linewidth=1,
                          edgecolor='#1A5276', facecolor='#AED6F1', zorder=3)
        ax.add_patch(rect)
        ax.text(tx + 0.4, 2.82, f'T{i}', ha='center', va='center',
                fontsize=7.5, fontweight='bold')

    ax.text(3.5, 3.25, 'Warp (32 threads)', ha='center', fontsize=9,
            fontweight='bold', color='#1A5276')

    # Scattered arrows (thread i → memory row i, showing non-sequential pattern)
    arrow_targets = [0, 4, 1, 6, 2, 7, 3, 5]
    for ti, mi in enumerate(arrow_targets):
        tx = ti * 0.95 + 0.4
        mx = mi * 0.95 + 0.4
        ax.annotate('', xy=(mx, 0.55), xytext=(tx, 2.6),
                    arrowprops=dict(arrowstyle='->', color='#E74C3C',
                                   lw=1.2, connectionstyle='arc3,rad=0.3'))

    ax.text(3.5, 1.6, '32 cache-line fetches\n(1 byte used each)', ha='center',
            fontsize=8.5, color='#C0392B', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FDEDEC', edgecolor='#E74C3C'))
    ax.text(3.5, 0.85, '19.2 GB/s  |  L2 hit 2.6%', ha='center', fontsize=9,
            color='#C0392B', fontweight='bold')

    # ─── Right: Warp-Opt INT8 ───
    ax = ax_warp
    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(-0.3, 4.0)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Stage 3: Warp-Opt INT8\n(Coalesced Access)', fontsize=11,
                 fontweight='bold', color='#1E8449', pad=8)

    # Consecutive memory block
    for i in range(8):
        mx = i * 0.95
        rect = Rectangle((mx, 0.1), 0.8, 0.45, linewidth=1.2,
                          edgecolor='#2C3E50', facecolor='#A9DFBF', zorder=2)
        ax.add_patch(rect)
        ax.text(mx + 0.4, 0.32, f'[{i*4}..{i*4+3}]', ha='center', va='center', fontsize=6)

    # 128-byte coalesced block annotation
    brace_rect = Rectangle((-0.1, 0.02), 7.85 + 0.8, 0.62,
                            linewidth=2, edgecolor='#1E8449',
                            facecolor='none', linestyle='--', zorder=3)
    ax.add_patch(brace_rect)
    ax.text(3.7, -0.2, '← 128-byte coalesced transaction →', ha='center',
            fontsize=8, color='#1E8449', style='italic')

    # Threads
    for i in range(8):
        tx = i * 0.95
        rect = Rectangle((tx, 2.6), 0.8, 0.45, linewidth=1,
                          edgecolor='#1A5276', facecolor='#AED6F1', zorder=3)
        ax.add_patch(rect)
        ax.text(tx + 0.4, 2.82, f'T{i}', ha='center', va='center',
                fontsize=7.5, fontweight='bold')

    ax.text(3.5, 3.25, 'Warp (32 threads)', ha='center', fontsize=9,
            fontweight='bold', color='#1A5276')

    # Parallel arrows (aligned)
    for i in range(8):
        tx = i * 0.95 + 0.4
        mx = i * 0.95 + 0.4
        ax.annotate('', xy=(mx, 0.55), xytext=(tx, 2.6),
                    arrowprops=dict(arrowstyle='->', color='#1E8449', lw=1.5))

    ax.text(3.5, 1.6, '1 cache-line fetch\n(128 bytes, all used)', ha='center',
            fontsize=8.5, color='#1E8449', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#EAFAF1', edgecolor='#1E8449'))
    ax.text(3.5, 0.85, '85.6 GB/s  |  5.7× speedup', ha='center', fontsize=9,
            color='#1E8449', fontweight='bold')

    plt.suptitle('Memory Access Pattern: Uncoalesced vs. Coalesced GEMV',
                 fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('Fig4_Memory_Access_Pattern.pdf', format='pdf', dpi=300, bbox_inches='tight')
    print("Done: Fig4_Memory_Access_Pattern.pdf")


# ─────────────────────────────────────────────
# Fig 5 (NEW): Energy Efficiency vs Throughput
# ─────────────────────────────────────────────
def plot_energy_efficiency():
    fig, ax1 = plt.subplots(figsize=(6.5, 4))

    stages = ['S1\nFP32', 'S2\nNaive\nINT8', 'S3\nWarp-Opt', 'S4\n+Fused\nOps',
              'S5\n+Flash\nAttn', 'S6\ndp4a\nW8A8']
    throughput = [14.9, 16.1, 31.8, 30.0, 35.2, 44.0]
    tok_per_j   = [None, None, None, None, 0.446, 0.584]

    x = np.arange(len(stages))
    colors = ['#888888', '#F5A623', '#4A90E2', '#7ED321', '#9013FE', '#D0021B']

    bars = ax1.bar(x, throughput, 0.45, color=colors, alpha=0.82, zorder=2,
                   label='Throughput (tok/s)')
    ax1.set_ylabel('Throughput (tok/s)', fontweight='bold')
    ax1.set_ylim(0, 55)
    ax1.set_xticks(x)
    ax1.set_xticklabels(stages, fontsize=8.5)
    ax1.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)

    # Bar value labels
    for bar, val in zip(bars, throughput):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.8,
                 f'{val}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # tok/J overlay
    ax2 = ax1.twinx()
    valid_x = [i for i, v in enumerate(tok_per_j) if v is not None]
    valid_y = [v for v in tok_per_j if v is not None]

    ax2.plot(valid_x, valid_y, color='#8B0000', marker='D', linewidth=2.5,
             markersize=10, label='Energy Efficiency (tok/J)', zorder=5)
    ax2.set_ylabel('Energy Efficiency (tok/J)', color='#8B0000', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#8B0000')
    ax2.set_ylim(0, 0.75)

    # Annotate improvement
    ax2.annotate('+31%\ntok/J', xy=(5, 0.584), xytext=(3.5, 0.67),
                 arrowprops=dict(arrowstyle='->', color='#8B0000', lw=1.8),
                 fontsize=9, color='#8B0000', fontweight='bold')
    ax2.text(4.55, 0.41, '0.446', fontsize=8.5, color='#8B0000', ha='center')
    ax2.text(5.45, 0.60, '0.584', fontsize=8.5, color='#8B0000', ha='center')

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper left', framealpha=0.9, fontsize=8.5)

    # Power note placed in bottom-right to avoid overlapping bars
    ax1.text(0.98, 0.04,
             'GPU power: 76.7 W ±4.2 W\n(stable across all stages)',
             transform=ax1.transAxes, fontsize=7.5, va='bottom', ha='right',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85, edgecolor='#bba'))

    plt.title('Throughput and Energy Efficiency Across Six Stages (RTX 4060 Laptop)',
              fontsize=10, pad=6)
    plt.tight_layout()
    plt.savefig('Fig5_Energy_Efficiency.pdf', format='pdf', dpi=300, bbox_inches='tight')
    print("Done: Fig5_Energy_Efficiency.pdf")


if __name__ == "__main__":
    print("Generating IEEE figures...")
    plot_memory_wall_ablation()
    plot_roofline_model()
    plot_context_scaling()
    plot_memory_access_diagram()
    plot_energy_efficiency()
    print("\nAll 5 figures generated.")
