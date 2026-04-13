# MiniLlama-Infra

A from-scratch CUDA inference engine for **Llama-3.2-1B** on consumer GPUs, built as a **micro-architecture ablation framework** for edge LLM deployment.

> **Paper:** *MiniLlama-Infra: A Micro-Architecture Ablation Study of Edge LLM Inference on Consumer GPUs*  
> Zijian Zhang, East China Normal University  
> [arXiv link — coming soon]

---

## Key Findings

This project is not about chasing SOTA throughput. The goal is to precisely measure *why* each optimization works at the hardware level using NVIDIA Nsight Compute profiling.

**Finding 1 — The Naive INT8 Trap:**  
Replacing FP32 weights with INT8 *without* changing the kernel layout makes things **worse**: memory throughput drops from 58.1 → 19.2 GB/s (−67%) due to catastrophic uncoalesced access. L2 hit rate collapses to 2.63%. Zero net speedup despite 4× smaller data.

**Finding 2 — Warp-Level Cooperation is the Critical Enabler:**  
Restructuring the kernel so 32 threads cooperatively compute one output row (coalesced loads + `__shfl_down_sync` warp reduce) delivers a **5.7× kernel speedup**, restoring GPU occupancy from 16.6% → 86.8%. This alone doubles end-to-end throughput from ~15 to ~31 tok/s.

**Finding 3 — Diminishing Returns After Coalescing:**  
Once the memory bottleneck is resolved, the system enters a compute-bound regime. Fusing Add+RMSNorm yields <1% improvement — the phase transition is the finding, not the optimization.

---

## Ablation Results (RTX 4060 Laptop, Llama-3.2-1B, d=2048)

| Stage | Kernel Duration | Mem. Throughput | Occupancy | End-to-End |
|-------|:-:|:-:|:-:|:-:|
| FP32 Baseline | 289.1 μs | 58.1 GB/s | 16.6% | 14.9 tok/s |
| Naive INT8 | 298.5 μs | 19.2 GB/s | 16.6% | 16.1 tok/s |
| Warp-Opt INT8 | **52.3 μs** | **85.6 GB/s** | **86.8%** | ~31 tok/s |
| + Fused Add+RMSNorm | — | — | 75.1% | ~31 tok/s |
| + Fused Attention | — | — | — | **35.2 tok/s** |

Full Nsight Compute metrics are in [`Float32_raw.csv`](Float32_raw.csv), [`naive_raw.csv`](naive_raw.csv), [`optimized_raw.csv`](optimized_raw.csv), [`fused_raw.csv`](fused_raw.csv).

---

## End-to-End Comparison

| Framework | Precision | Peak VRAM | Throughput | WikiText-2 PPL |
|-----------|:-:|:-:|:-:|:-:|
| PyTorch (HuggingFace) | FP16 | 2377 MB | 59.3 tok/s | 11.50 |
| llama.cpp | Q8_0 (global) | 1252 MB | 144.9 tok/s | 11.80 |
| **MiniLlama-Infra (ours)** | **INT8+FP32** | **2054 MB** | **35.2 tok/s** | **11.52** |

Our hybrid-precision strategy (INT8 for hidden layers, FP32 for vocabulary embedding) achieves 0.28 PPL lower than llama.cpp's global Q8_0, at 13.6% lower VRAM than PyTorch.

---

## System Architecture

![System Architecture](Fig0_Architecture_v2.pdf)

Three optimization points:
- **Red** — Warp-level cooperative INT8 GEMV with on-the-fly dequantization
- **Blue** — Fused Add+RMSNorm kernel (eliminates one global memory round-trip)
- **Green** — Fused attention (QK·Softmax·V in shared memory, reduces 560 kernel launches to 16)

---

## Project Structure

```
├── src/
│   ├── model_cuda_v2_shared.cu   # All CUDA kernels (GEMV, attention, norms, fusion)
│   ├── main_cuda.cpp             # Inference engine, benchmark harness, context scaling
│   └── tokenizer.cpp             # BPE tokenizer
├── include/
│   ├── config.h                  # Model config (dims, layers, vocab size)
│   ├── model.h                   # GPU state structs, weight layout
│   └── tokenizer.h
├── quantize_int8_group.py        # Convert HuggingFace weights → INT8 group-64 .bin
├── eval_ppl.py                   # WikiText-2 perplexity evaluation (simulated quantization)
├── benchmark_hf.py               # PyTorch baseline benchmark
├── plot_paper_figs.py            # Reproduce all paper figures from CSV data
├── *_raw.csv                     # Raw Nsight Compute exports for each ablation stage
└── conference_101719.tex         # Paper source (IEEEtran)
```

---

## Build

**Requirements:** CUDA 12+, Visual Studio (Windows) or GCC (Linux), NVCC

```bash
# Windows (from project root, in Developer Command Prompt)
nvcc -o build/mini_llama_infra.exe src/main_cuda.cpp src/model_cuda_v2_shared.cu src/tokenizer.cpp \
     -Xcompiler "/utf-8" -arch=sm_89

# Linux (replace -Xcompiler flag)
nvcc -o build/mini_llama_infra src/main_cuda.cpp src/model_cuda_v2_shared.cu src/tokenizer.cpp \
     -arch=sm_89
```

The `-arch=sm_89` flag targets Ada Lovelace (RTX 4060/4070/4080/4090). Change to `sm_86` for Ampere (RTX 3xxx).

---

## Prepare Model Weights

You need the Llama-3.2-1B weights in our custom binary format.

```bash
# 1. Download the HuggingFace model
python downloadmodel.py   # requires HF token with Llama-3.2 access

# 2. Quantize to INT8 group-64 format
python quantize_int8_group.py
# Outputs: llama3_2_1B_q8_group.bin (~1.3 GB)
# Also needs: tokenizer.bin (copy from HF model files)
```

---

## Run

```bash
# Run benchmark (4 prompts, 128 tokens each)
./build/mini_llama_infra

# Expected output:
# Peak VRAM: 2054 MB
# Throughput: ~35 tok/s
```

---

## Reproduce Paper Figures

```bash
# Requires: matplotlib, pandas
python plot_paper_figs.py
# Generates Fig1–Fig3 from the raw Nsight CSV exports
```

---

## Perplexity Evaluation

```bash
# Requires: transformers, torch, datasets
# Uses simulated (fake) INT8 quantization on HuggingFace model
python eval_ppl.py
# Expected: FP16 baseline 11.50 | INT8 group-64 11.52
```

---

## Hardware Notes

All profiling was done on **NVIDIA GeForce RTX 4060 Laptop GPU** (Ada Lovelace, SM 8.9, 8 GB GDDR6, 272 GB/s peak bandwidth). Kernel-level metrics may differ on other architectures but the memory access patterns generalize across consumer GPUs.

---

## Citation

If you use this code or findings, please cite:

```bibtex
@article{zhang2025minillama,
  title={MiniLlama-Infra: A Micro-Architecture Ablation Study of Edge LLM Inference on Consumer GPUs},
  author={Zhang, Zijian},
  journal={arXiv preprint},
  year={2025}
}
```
