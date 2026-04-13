import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9

def plot_memory_wall_ablation():
    """Fig 1: Ablation - Memory Throughput + Decoding Speed (5 stages)"""
    fig, ax1 = plt.subplots(figsize=(7, 4))
    
    labels = ['FP32\nBaseline', 'Naive\nINT8', 'Warp-Opt\nINT8', '+Fused\nOps', '+Flash\nAttn']
    mem_throughput = [58.08, 19.20, 85.57, 85.57, 85.57]  # matmul throughput unchanged for last two
    decoding_speed = [14.9, 16.1, 31.8, 30.0, 35.0]

    x = np.arange(len(labels))
    width = 0.4

    bars = ax1.bar(x, mem_throughput, width, color='#4A90E2', alpha=0.85, label='MatMul Memory Throughput (GB/s)')
    ax1.set_ylabel('MatMul Memory Throughput (GB/s)', color='#4A90E2', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#4A90E2')
    ax1.set_ylim(0, 110)

    ax2 = ax1.twinx()
    ax2.plot(x, decoding_speed, color='#D0021B', marker='o', linewidth=2.5, markersize=8, label='Decoding Speed (tok/s)', zorder=5)
    ax2.set_ylabel('Decoding Speed (tok/s)', color='#D0021B', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#D0021B')
    ax2.set_ylim(0, 45)

    # Annotate key points
    ax2.annotate('Naive INT8\nslower than FP32!', xy=(1, 16.1), xytext=(1.5, 8),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=9, color='red', fontstyle='italic')
    ax2.annotate('+107%', xy=(2, 31.8), xytext=(2.3, 38),
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
                fontsize=10, color='green', fontweight='bold')

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', framealpha=0.9)

    plt.tight_layout()
    plt.savefig('Fig1_Memory_Wall_Ablation.pdf', format='pdf', dpi=300, bbox_inches='tight')
    print("Done: Fig1_Memory_Wall_Ablation.pdf")

def plot_roofline_model():
    """Fig 2: Roofline Model"""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    mem_bandwidth = 272.0  # RTX 4060 Laptop peak BW GB/s

    points = {
        'FP32 Baseline': (0.5, 29.0, '#888888', 's'),
        'Naive INT8':    (2.0, 28.0, '#F5A623', 'v'),
        'Warp-Opt INT8': (2.0, 160.0, '#50E3C2', '^')
    }

    x_line = np.logspace(-1, 1.5, 100)
    y_bw = mem_bandwidth * x_line
    ax.plot(x_line, y_bw, color='black', linestyle='--', linewidth=1.5, 
            label=f'Peak Memory BW ({mem_bandwidth} GB/s)')
    
    for label, (intensity, perf, color, marker) in points.items():
        ax.scatter(intensity, perf, color=color, marker=marker, s=150, 
                  label=label, zorder=5, edgecolors='black', linewidths=0.5)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.2, 10)
    ax.set_ylim(10, 1000)
    
    ax.set_xlabel('Arithmetic Intensity (FLOPs/Byte)', fontweight='bold')
    ax.set_ylabel('Performance (GOPS)', fontweight='bold')
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend(loc='lower right', framealpha=0.9)

    plt.tight_layout()
    plt.savefig('Fig2_Roofline_Model.pdf', format='pdf', dpi=300, bbox_inches='tight')
    print("Done: Fig2_Roofline_Model.pdf")

def plot_context_scaling():
    """Fig 3: Context Scaling TTFT"""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    context_lengths = [53, 85, 149, 277, 533]
    ttft_seconds = [1.604, 2.521, 4.427, 8.188, 15.792]

    ax.plot(context_lengths, ttft_seconds, marker='D', color='#9013FE', 
            linewidth=2.5, markersize=8, label='MiniLlama-Infra')
    ax.fill_between(context_lengths, ttft_seconds, color='#9013FE', alpha=0.08)

    # Add ideal linear reference line
    scale = ttft_seconds[0] / context_lengths[0]
    ideal_line = [l * scale for l in context_lengths]
    ax.plot(context_lengths, ideal_line, '--', color='gray', linewidth=1, 
            alpha=0.7, label='Ideal O(N) reference')

    ax.set_xlabel('Context Length (Tokens)', fontweight='bold')
    ax.set_ylabel('Time to First Token (Seconds)', fontweight='bold')
    
    ax.set_xticks(context_lengths)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    ax.annotate('Sequential GEMV\nMemory-Bound', 
                xy=(277, 8.188), xytext=(100, 11),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6),
                fontsize=10)

    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('Fig3_Context_Scaling.pdf', format='pdf', dpi=300, bbox_inches='tight')
    print("Done: Fig3_Context_Scaling.pdf")

if __name__ == "__main__":
    print("Generating IEEE figures...")
    plot_memory_wall_ablation()
    plot_roofline_model()
    plot_context_scaling()
    print("All figures generated.")
