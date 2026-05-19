#include "../include/model.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <algorithm>

// ====================================================================
// 🚀 CUDA Kernel: 标准 FP32 矩阵乘法
// ====================================================================
__global__ void matmul_kernel(float* xout, float* x, float* w, int n, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < d) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

void matmul(float* xout, float* x, float* w, int n, int d) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (d + threadsPerBlock - 1) / threadsPerBlock;
    matmul_kernel<<<blocksPerGrid, threadsPerBlock>>>(xout, x, w, n, d);
}

// ====================================================================
// 👑 Warp-Level INT8 Group MatMul
// ====================================================================
__device__ inline float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void matmul_int8_group_kernel_v2(float* xout, float* x, int8_t* w_q, float* w_s, int n, int d, int group_size) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;
    int row = warp_id;

    if (row < d) {
        float val = 0.0f;
        int groups_per_row = n / group_size;

        for (int g = 0; g < groups_per_row; g++) {
            float scale = w_s[row * groups_per_row + g];
            int start_idx = row * n + g * group_size;

            for (int j = lane_id; j < group_size; j += 32) {
                int col = g * group_size + j;
                float weight_fp32 = (float)w_q[start_idx + j] * scale;
                val += weight_fp32 * x[col];
            }
        }

        val = warpReduceSum(val);

        if (lane_id == 0) {
            xout[row] = val;
        }
    }
}

void matmul_int8(float* xout, float* x, int8_t* w_q, float* w_s, int n, int d, int group_size) {
    int threadsPerBlock = 256;
    int rowsPerBlock = threadsPerBlock / 32;
    int blocksPerGrid = (d + rowsPerBlock - 1) / rowsPerBlock;
    matmul_int8_group_kernel_v2<<<blocksPerGrid, threadsPerBlock>>>(xout, x, w_q, w_s, n, d, group_size);
}

// ====================================================================
// 🚀 RMSNorm (独立版)
// ====================================================================
__global__ void rmsnorm_kernel(float* o, float* x, float* weight, int size) {
    __shared__ float s_sum;
    int tid = threadIdx.x;
    if (tid == 0) s_sum = 0.0f;
    __syncthreads();
    float local_sum = 0.0f;
    for (int i = tid; i < size; i += blockDim.x) {
        local_sum += x[i] * x[i];
    }
    atomicAdd(&s_sum, local_sum);
    __syncthreads();
    float ss = 1.0f / sqrtf(s_sum / size + 1e-5f);
    for (int i = tid; i < size; i += blockDim.x) {
        o[i] = weight[i] * (ss * x[i]);
    }
}

void rmsnorm_cuda(float* o, float* x, float* weight, int size) {
    int threads = (size < 1024) ? size : 1024;
    rmsnorm_kernel<<<1, threads>>>(o, x, weight, size);
}

// ====================================================================
// 🔥 融合算子: Add + RMSNorm
// ====================================================================
__global__ void fused_add_rmsnorm_kernel(float* o, float* x, float* residual, float* weight, int size) {
    __shared__ float s_sum;
    int tid = threadIdx.x;
    if (tid == 0) s_sum = 0.0f;
    __syncthreads();

    float local_sum = 0.0f;
    for (int i = tid; i < size; i += blockDim.x) {
        float val = x[i] + residual[i];
        x[i] = val;
        local_sum += val * val;
    }
    atomicAdd(&s_sum, local_sum);
    __syncthreads();

    float ss = 1.0f / sqrtf(s_sum / size + 1e-5f);
    for (int i = tid; i < size; i += blockDim.x) {
        o[i] = weight[i] * (ss * x[i]);
    }
}

void fused_add_rmsnorm_cuda(float* o, float* x, float* residual, float* weight, int size) {
    int threads = (size < 1024) ? size : 1024;
    fused_add_rmsnorm_kernel<<<1, threads>>>(o, x, residual, weight, size);
}

// ====================================================================
// 📦 GPU VRAM Management (INT8 Group-wise)
// ====================================================================
void alloc_gpu_weights(TransformerWeightsGPU* w_gpu, TransformerWeights* w_cpu, Config* p) {
    int dim = p->dim; int hidden_dim = p->hidden_dim; int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int gs = p->group_size;

    cudaMalloc((void**)&w_gpu->token_embedding_table, p->vocab_size * dim * 4);
    cudaMalloc((void**)&w_gpu->rms_att_weight, p->n_layers * dim * 4);
    
    int wq_elems = p->n_layers * dim * dim;
    cudaMalloc((void**)&w_gpu->wq_s, (wq_elems / gs) * 4); cudaMalloc((void**)&w_gpu->wq_q, wq_elems * 1);
    int wk_elems = p->n_layers * dim * kv_dim;
    cudaMalloc((void**)&w_gpu->wk_s, (wk_elems / gs) * 4); cudaMalloc((void**)&w_gpu->wk_q, wk_elems * 1);
    int wv_elems = p->n_layers * dim * kv_dim;
    cudaMalloc((void**)&w_gpu->wv_s, (wv_elems / gs) * 4); cudaMalloc((void**)&w_gpu->wv_q, wv_elems * 1);
    int wo_elems = p->n_layers * dim * dim;
    cudaMalloc((void**)&w_gpu->wo_s, (wo_elems / gs) * 4); cudaMalloc((void**)&w_gpu->wo_q, wo_elems * 1);

    cudaMalloc((void**)&w_gpu->rms_ffn_weight, p->n_layers * dim * 4);

    int w1_elems = p->n_layers * dim * hidden_dim;
    cudaMalloc((void**)&w_gpu->w1_s, (w1_elems / gs) * 4); cudaMalloc((void**)&w_gpu->w1_q, w1_elems * 1);
    int w2_elems = p->n_layers * hidden_dim * dim;
    cudaMalloc((void**)&w_gpu->w2_s, (w2_elems / gs) * 4); cudaMalloc((void**)&w_gpu->w2_q, w2_elems * 1);
    int w3_elems = p->n_layers * dim * hidden_dim;
    cudaMalloc((void**)&w_gpu->w3_s, (w3_elems / gs) * 4); cudaMalloc((void**)&w_gpu->w3_q, w3_elems * 1);

    cudaMalloc((void**)&w_gpu->rms_final_weight, dim * 4);
    
    cudaMemcpy(w_gpu->token_embedding_table, w_cpu->token_embedding_table, p->vocab_size * dim * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(w_gpu->rms_att_weight, w_cpu->rms_att_weight, p->n_layers * dim * 4, cudaMemcpyHostToDevice);
    
    cudaMemcpy(w_gpu->wq_s, w_cpu->wq_s, (wq_elems / gs) * 4, cudaMemcpyHostToDevice); cudaMemcpy(w_gpu->wq_q, w_cpu->wq_q, wq_elems * 1, cudaMemcpyHostToDevice);
    cudaMemcpy(w_gpu->wk_s, w_cpu->wk_s, (wk_elems / gs) * 4, cudaMemcpyHostToDevice); cudaMemcpy(w_gpu->wk_q, w_cpu->wk_q, wk_elems * 1, cudaMemcpyHostToDevice);
    cudaMemcpy(w_gpu->wv_s, w_cpu->wv_s, (wv_elems / gs) * 4, cudaMemcpyHostToDevice); cudaMemcpy(w_gpu->wv_q, w_cpu->wv_q, wv_elems * 1, cudaMemcpyHostToDevice);
    cudaMemcpy(w_gpu->wo_s, w_cpu->wo_s, (wo_elems / gs) * 4, cudaMemcpyHostToDevice); cudaMemcpy(w_gpu->wo_q, w_cpu->wo_q, wo_elems * 1, cudaMemcpyHostToDevice);

    cudaMemcpy(w_gpu->rms_ffn_weight, w_cpu->rms_ffn_weight, p->n_layers * dim * 4, cudaMemcpyHostToDevice);

    cudaMemcpy(w_gpu->w1_s, w_cpu->w1_s, (w1_elems / gs) * 4, cudaMemcpyHostToDevice); cudaMemcpy(w_gpu->w1_q, w_cpu->w1_q, w1_elems * 1, cudaMemcpyHostToDevice);
    cudaMemcpy(w_gpu->w2_s, w_cpu->w2_s, (w2_elems / gs) * 4, cudaMemcpyHostToDevice); cudaMemcpy(w_gpu->w2_q, w_cpu->w2_q, w2_elems * 1, cudaMemcpyHostToDevice);
    cudaMemcpy(w_gpu->w3_s, w_cpu->w3_s, (w3_elems / gs) * 4, cudaMemcpyHostToDevice); cudaMemcpy(w_gpu->w3_q, w_cpu->w3_q, w3_elems * 1, cudaMemcpyHostToDevice);

    cudaMemcpy(w_gpu->rms_final_weight, w_cpu->rms_final_weight, dim * 4, cudaMemcpyHostToDevice);
    
    w_gpu->wcls = w_gpu->token_embedding_table;
    cudaDeviceSynchronize();
}

void free_gpu_weights(TransformerWeightsGPU* w) {
    cudaFree(w->token_embedding_table); cudaFree(w->rms_att_weight);
    cudaFree(w->wq_q); cudaFree(w->wq_s); cudaFree(w->wk_q); cudaFree(w->wk_s);
    cudaFree(w->wv_q); cudaFree(w->wv_s); cudaFree(w->wo_q); cudaFree(w->wo_s);
    cudaFree(w->rms_ffn_weight);
    cudaFree(w->w1_q); cudaFree(w->w1_s); cudaFree(w->w2_q); cudaFree(w->w2_s);
    cudaFree(w->w3_q); cudaFree(w->w3_s); cudaFree(w->rms_final_weight);
}

void alloc_gpu_state(RunStateGPU* s_gpu, Config* p) {
    int dim = p->dim; int hidden_dim = p->hidden_dim; int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    cudaMalloc((void**)&s_gpu->x, dim * sizeof(float));
    cudaMalloc((void**)&s_gpu->xb, dim * sizeof(float));
    cudaMalloc((void**)&s_gpu->xb2, dim * sizeof(float));
    cudaMalloc((void**)&s_gpu->hb, hidden_dim * sizeof(float));
    cudaMalloc((void**)&s_gpu->hb2, hidden_dim * sizeof(float));
    cudaMalloc((void**)&s_gpu->q, dim * sizeof(float));
    cudaMalloc((void**)&s_gpu->k, dim * sizeof(float));
    cudaMalloc((void**)&s_gpu->v, dim * sizeof(float));
    cudaMalloc((void**)&s_gpu->att, p->n_heads * p->seq_len * sizeof(float));
    cudaMalloc((void**)&s_gpu->logits, p->vocab_size * sizeof(float));
    cudaMalloc((void**)&s_gpu->key_cache, p->n_layers * p->seq_len * kv_dim * sizeof(float));
    cudaMalloc((void**)&s_gpu->value_cache, p->n_layers * p->seq_len * kv_dim * sizeof(float));
    int act_buf_size = (hidden_dim > dim) ? hidden_dim : dim;
    cudaMalloc((void**)&s_gpu->act_q, act_buf_size * sizeof(int8_t));
    cudaMalloc((void**)&s_gpu->act_s, sizeof(float));
}

void free_gpu_state(RunStateGPU* s_gpu) {
    cudaFree(s_gpu->x); cudaFree(s_gpu->xb); cudaFree(s_gpu->xb2);
    cudaFree(s_gpu->hb); cudaFree(s_gpu->hb2);
    cudaFree(s_gpu->q); cudaFree(s_gpu->k); cudaFree(s_gpu->v);
    cudaFree(s_gpu->att); cudaFree(s_gpu->logits);
    cudaFree(s_gpu->key_cache); cudaFree(s_gpu->value_cache);
    cudaFree(s_gpu->act_q); cudaFree(s_gpu->act_s);
}

// ====================================================================
// 🔧 Supplementary Kernels
// ====================================================================
__global__ void add_kernel(float* x, float* y, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) x[i] += y[i];
}

__global__ void silu_mul_kernel(float* hb, float* hb2, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float val = hb[i];
        val *= (1.0f / (1.0f + expf(-val)));
        val *= hb2[i];
        hb[i] = val;
    }
}

// ====================================================================
// 🚀 Llama 3.2 RoPE
// ====================================================================
__device__ float get_llama3_scaled_freq(int head_dim_idx, int head_size) {
    float freq = 1.0f / powf(500000.0f, (head_dim_idx * 2) / (float)head_size);
    float wavelength = 2.0f * 3.141592653589793f / freq;
    
    float factor = 32.0f;
    float low_freq_factor = 1.0f;
    float high_freq_factor = 4.0f;
    float old_context_len = 8192.0f;
    
    if (wavelength > old_context_len / low_freq_factor) {
        freq = freq / factor;
    } else if (wavelength >= old_context_len / high_freq_factor) {
        float smooth = (old_context_len / wavelength - low_freq_factor) / (high_freq_factor - low_freq_factor);
        freq = (1.0f - smooth) * (freq / factor) + smooth * freq;
    }
    return freq;
}

__global__ void rope_kernel(float* q, float* k, int pos, int dim, int kv_dim, int head_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < dim) {
        int head_dim_idx = i % head_size;
        if (head_dim_idx < head_size / 2) {
            float freq = get_llama3_scaled_freq(head_dim_idx, head_size);
            float val = pos * freq;
            float fcr = cosf(val), fci = sinf(val);
            int pair_idx = i + head_size / 2;
            
            float q0 = q[i], q1 = q[pair_idx];
            q[i]        = q0 * fcr - q1 * fci;
            q[pair_idx] = q0 * fci + q1 * fcr;
        }
    }
    
    if (i < kv_dim) {
        int head_dim_idx = i % head_size;
        if (head_dim_idx < head_size / 2) {
            float freq = get_llama3_scaled_freq(head_dim_idx, head_size);
            float val = pos * freq;
            float fcr = cosf(val), fci = sinf(val);
            int pair_idx = i + head_size / 2;
            
            float k0 = k[i], k1 = k[pair_idx];
            k[i]        = k0 * fcr - k1 * fci;
            k[pair_idx] = k0 * fci + k1 * fcr;
        }
    }
}

// ====================================================================
// 🔥 Fused Attention Kernel (Flash Attention 风格)
// 将 QK^T + Softmax + V*Att 三步融合为一个 Kernel
// 原来: attention_query_key_kernel (1次) + cudaDeviceSynchronize
//       + softmax_cuda (32次/层) + attention_value_kernel (1次)
//     = 每层 34 次 kernel launch + 1 次同步
// 现在: 每层只需 1 次 kernel launch，零同步
// ====================================================================
__global__ void fused_attention_kernel(
    float* xb,            // 输出 [dim]
    float* q,             // Query [dim]
    float* key_cache,     // KV Cache
    float* value_cache,   // KV Cache
    int head_size, int kv_dim, int kv_mul, int seq_len, int loff, int pos
) {
    int h = blockIdx.x;   // 每个 block 处理一个 attention head
    int tid = threadIdx.x;
    int kv_head = h / kv_mul;
    float* h_q = q + h * head_size;

    // 动态共享内存: 前 (pos+1) 个 float 存 attention scores
    extern __shared__ float shared_att[];

    // ==========================================
    // Step 1: 计算 QK^T (scores 存入 shared memory，不写全局显存)
    // ==========================================
    for (int t = tid; t <= pos; t += blockDim.x) {
        float* h_k = key_cache + loff + t * kv_dim + kv_head * head_size;
        float score = 0.0f;
        for (int i = 0; i < head_size; i++) {
            score += h_q[i] * h_k[i];
        }
        shared_att[t] = score / sqrtf((float)head_size);
    }
    __syncthreads();

    // ==========================================
    // Step 2: Softmax (完全在 shared memory 内完成)
    // ==========================================
    // 2a. 并行求 max
    __shared__ float s_max;
    if (tid == 0) {
        float max_val = shared_att[0];
        for (int t = 1; t <= pos; t++) {
            if (shared_att[t] > max_val) max_val = shared_att[t];
        }
        s_max = max_val;
    }
    __syncthreads();

    // 2b. 并行 exp
    for (int t = tid; t <= pos; t += blockDim.x) {
        shared_att[t] = expf(shared_att[t] - s_max);
    }
    __syncthreads();

    // 2c. 并行求 sum
    __shared__ float s_sum;
    if (tid == 0) {
        float sum = 0.0f;
        for (int t = 0; t <= pos; t++) sum += shared_att[t];
        s_sum = sum;
    }
    __syncthreads();

    // 2d. 归一化
    for (int t = tid; t <= pos; t += blockDim.x) {
        shared_att[t] /= s_sum;
    }
    __syncthreads();

    // ==========================================
    // Step 3: 加权求和 V (直接输出，无需中间 buffer)
    // ==========================================
    if (tid < head_size) {
        float val = 0.0f;
        for (int t = 0; t <= pos; t++) {
            float* h_v = value_cache + loff + t * kv_dim + kv_head * head_size;
            val += shared_att[t] * h_v[tid];
        }
        xb[h * head_size + tid] = val;
    }
}

// ====================================================================
// 🚀 Forward Pass (INT8 + 融合算子 + Flash Attention)
// ====================================================================
float* forward_cuda(int token, int pos, Config* p, TransformerWeightsGPU* w, RunStateGPU* s) {
    int dim = p->dim; int hidden_dim = p->hidden_dim; int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; int head_size = dim / p->n_heads;
    int gs = p->group_size;
    int threads = 256;
    int blocks_dim = (dim + threads - 1) / threads;
    int blocks_hidden = (hidden_dim + threads - 1) / threads;

    cudaMemcpy(s->x, w->token_embedding_table + token * dim, dim * sizeof(float), cudaMemcpyDeviceToDevice);

    for (int l = 0; l < p->n_layers; l++) {
        rmsnorm_cuda(s->xb, s->x, w->rms_att_weight + l * dim, dim);
        
        // Q K V MatMul (INT8)
        matmul_int8(s->q, s->xb, w->wq_q + l * dim * dim, w->wq_s + l * (dim * dim / gs), dim, dim, gs);
        matmul_int8(s->k, s->xb, w->wk_q + l * dim * kv_dim, w->wk_s + l * (dim * kv_dim / gs), dim, kv_dim, gs);
        matmul_int8(s->v, s->xb, w->wv_q + l * dim * kv_dim, w->wv_s + l * (dim * kv_dim / gs), dim, kv_dim, gs);

        rope_kernel<<<blocks_dim, threads>>>(s->q, s->k, pos, dim, kv_dim, head_size);

        int loff = l * p->seq_len * kv_dim;
        cudaMemcpy(s->key_cache + loff + pos * kv_dim, s->k, kv_dim * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(s->value_cache + loff + pos * kv_dim, s->v, kv_dim * sizeof(float), cudaMemcpyDeviceToDevice);

        // 🔥 Flash Attention: 1 次 launch 替代原来的 34 次 + 1 次 sync
        // 共享内存大小 = (pos+1) * sizeof(float)，最大 1024*4 = 4KB
        size_t smem_size = (pos + 1) * sizeof(float);
        fused_attention_kernel<<<p->n_heads, 256, smem_size>>>(
            s->xb, s->q, s->key_cache, s->value_cache,
            head_size, kv_dim, kv_mul, p->seq_len, loff, pos
        );
        
        matmul_int8(s->xb2, s->xb, w->wo_q + l * dim * dim, w->wo_s + l * (dim * dim / gs), dim, dim, gs);

        // 🔥 融合 Add + RMSNorm
        fused_add_rmsnorm_cuda(s->xb, s->x, s->xb2, w->rms_ffn_weight + l * dim, dim);

        // FFN
        matmul_int8(s->hb, s->xb, w->w1_q + l * dim * hidden_dim, w->w1_s + l * (dim * hidden_dim / gs), dim, hidden_dim, gs);
        matmul_int8(s->hb2, s->xb, w->w3_q + l * dim * hidden_dim, w->w3_s + l * (dim * hidden_dim / gs), dim, hidden_dim, gs);
        silu_mul_kernel<<<blocks_hidden, threads>>>(s->hb, s->hb2, hidden_dim);
        
        matmul_int8(s->xb, s->hb, w->w2_q + l * hidden_dim * dim, w->w2_s + l * (hidden_dim * dim / gs), hidden_dim, dim, gs);

        add_kernel<<<blocks_dim, threads>>>(s->x, s->xb, dim);
    }

    rmsnorm_cuda(s->x, s->x, w->rms_final_weight, dim);
    matmul(s->logits, s->x, w->wcls, dim, p->vocab_size);

    return s->logits;
}

// ====================================================================
// 🚀 Stage 6: dp4a W8A8 Kernels
// ====================================================================

// Per-token dynamic INT8 quantization of activation vector.
// Uses float-as-uint atomicMax trick (valid for non-negative floats).
__global__ void quantize_activation_kernel(int8_t* xq, float* xs, const float* x, int n) {
    __shared__ unsigned int s_max_bits;
    if (threadIdx.x == 0) s_max_bits = 0;
    __syncthreads();

    float local_max = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        local_max = fmaxf(local_max, fabsf(x[i]));

    for (int offset = 16; offset > 0; offset /= 2)
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));

    if (threadIdx.x % 32 == 0)
        atomicMax(&s_max_bits, __float_as_uint(local_max));
    __syncthreads();

    float scale = __uint_as_float(s_max_bits) / 127.0f;
    if (scale < 1e-8f) scale = 1e-8f;
    if (threadIdx.x == 0) *xs = scale;
    __syncthreads();

    float inv_scale = 1.0f / scale;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = x[i] * inv_scale;
        xq[i] = (int8_t)max(-127, min(127, __float2int_rn(v)));
    }
}

void quantize_activation(int8_t* xq, float* xs, const float* x, int n) {
    int threads = (n < 1024) ? n : 1024;
    quantize_activation_kernel<<<1, threads>>>(xq, xs, x, n);
}

// Warp-level dp4a W8A8 GEMV.
// Each warp owns one output row; __dp4a accumulates 4 INT8 MACs per cycle.
// group_size must be divisible by 4 (satisfied for group_size=64).
__global__ void matmul_dp4a_kernel(float* xout,
                                    const int8_t* __restrict__ x_q, const float* x_s,
                                    const int8_t* __restrict__ w_q, const float* w_s,
                                    int n, int d, int group_size) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;
    int row = warp_id;
    if (row >= d) return;

    float x_scale = *x_s;
    int groups_per_row = n / group_size;
    float fp32_acc = 0.0f;

    for (int g = 0; g < groups_per_row; g++) {
        float w_scale = w_s[row * groups_per_row + g];
        int w_base = row * n + g * group_size;
        int x_base = g * group_size;

        int32_t group_acc = 0;
        // Each lane processes 4 elements at a time; stride = 32*4 = 128.
        // For group_size=64: only lanes 0-15 execute the body (j < 64).
        for (int j = lane_id * 4; j < group_size; j += 128) {
            int w_pack = *reinterpret_cast<const int32_t*>(&w_q[w_base + j]);
            int x_pack = *reinterpret_cast<const int32_t*>(&x_q[x_base + j]);
            group_acc = __dp4a(w_pack, x_pack, group_acc);
        }

        // Warp-level INT32 reduce
        for (int offset = 16; offset > 0; offset /= 2)
            group_acc += __shfl_down_sync(0xffffffff, group_acc, offset);

        if (lane_id == 0)
            fp32_acc += (float)group_acc * w_scale * x_scale;
    }

    if (lane_id == 0) xout[row] = fp32_acc;
}

void matmul_dp4a(float* xout, const int8_t* x_q, const float* x_s,
                 const int8_t* w_q, const float* w_s, int n, int d, int group_size) {
    int threadsPerBlock = 256;
    int rowsPerBlock = threadsPerBlock / 32;
    int blocksPerGrid = (d + rowsPerBlock - 1) / rowsPerBlock;
    matmul_dp4a_kernel<<<blocksPerGrid, threadsPerBlock>>>(xout, x_q, x_s, w_q, w_s, n, d, group_size);
}

// ====================================================================
// 🚀 Stage 6 Forward Pass (dp4a W8A8)
// ====================================================================
float* forward_cuda_dp4a(int token, int pos, Config* p, TransformerWeightsGPU* w, RunStateGPU* s) {
    int dim = p->dim; int hidden_dim = p->hidden_dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; int head_size = dim / p->n_heads;
    int gs = p->group_size;
    int threads = 256;
    int blocks_dim = (dim + threads - 1) / threads;
    int blocks_hidden = (hidden_dim + threads - 1) / threads;

    cudaMemcpy(s->x, w->token_embedding_table + token * dim, dim * sizeof(float), cudaMemcpyDeviceToDevice);

    for (int l = 0; l < p->n_layers; l++) {
        rmsnorm_cuda(s->xb, s->x, w->rms_att_weight + l * dim, dim);

        // Quantize xb → act_q/act_s, then dp4a for Q K V
        quantize_activation(s->act_q, s->act_s, s->xb, dim);
        matmul_dp4a(s->q,   s->act_q, s->act_s, w->wq_q + l*dim*dim,     w->wq_s + l*(dim*dim/gs),     dim, dim,    gs);
        matmul_dp4a(s->k,   s->act_q, s->act_s, w->wk_q + l*dim*kv_dim,  w->wk_s + l*(dim*kv_dim/gs),  dim, kv_dim, gs);
        matmul_dp4a(s->v,   s->act_q, s->act_s, w->wv_q + l*dim*kv_dim,  w->wv_s + l*(dim*kv_dim/gs),  dim, kv_dim, gs);

        rope_kernel<<<blocks_dim, threads>>>(s->q, s->k, pos, dim, kv_dim, head_size);

        int loff = l * p->seq_len * kv_dim;
        cudaMemcpy(s->key_cache + loff + pos*kv_dim, s->k, kv_dim*sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(s->value_cache + loff + pos*kv_dim, s->v, kv_dim*sizeof(float), cudaMemcpyDeviceToDevice);

        size_t smem = (pos + 1) * sizeof(float);
        fused_attention_kernel<<<p->n_heads, 256, smem>>>(s->xb, s->q, s->key_cache, s->value_cache,
                                                           head_size, kv_dim, kv_mul, p->seq_len, loff, pos);

        // Quantize attention output (xb) for output projection
        quantize_activation(s->act_q, s->act_s, s->xb, dim);
        matmul_dp4a(s->xb2, s->act_q, s->act_s, w->wo_q + l*dim*dim, w->wo_s + l*(dim*dim/gs), dim, dim, gs);

        fused_add_rmsnorm_cuda(s->xb, s->x, s->xb2, w->rms_ffn_weight + l*dim, dim);

        // Quantize for FFN gate/up projections
        quantize_activation(s->act_q, s->act_s, s->xb, dim);
        matmul_dp4a(s->hb,  s->act_q, s->act_s, w->w1_q + l*dim*hidden_dim, w->w1_s + l*(dim*hidden_dim/gs), dim, hidden_dim, gs);
        matmul_dp4a(s->hb2, s->act_q, s->act_s, w->w3_q + l*dim*hidden_dim, w->w3_s + l*(dim*hidden_dim/gs), dim, hidden_dim, gs);
        silu_mul_kernel<<<blocks_hidden, threads>>>(s->hb, s->hb2, hidden_dim);

        // Quantize hb for FFN down projection
        quantize_activation(s->act_q, s->act_s, s->hb, hidden_dim);
        matmul_dp4a(s->xb, s->act_q, s->act_s, w->w2_q + l*hidden_dim*dim, w->w2_s + l*(hidden_dim*dim/gs), hidden_dim, dim, gs);

        add_kernel<<<blocks_dim, threads>>>(s->x, s->xb, dim);
    }

    rmsnorm_cuda(s->x, s->x, w->rms_final_weight, dim);
    matmul(s->logits, s->x, w->wcls, dim, p->vocab_size);
    return s->logits;
}

// ====================================================================
// 🎲 Helper & Sampler
// ====================================================================
int argmax(float* logits, int size) {
    int max_i = 0; float max_val = logits[0];
    for (int i = 1; i < size; i++) if (logits[i] > max_val) { max_val = logits[i]; max_i = i; }
    return max_i;
}

struct ProbIndex { float prob; int index; };
bool compareProbIndex(const ProbIndex& a, const ProbIndex& b) { return a.prob > b.prob; }

int sample_topp(float* logits, int size, float temperature, float topp) {
    if (temperature < 1e-6f) return argmax(logits, size);
    float max_val = logits[0] / temperature;
    for (int i = 1; i < size; i++) {
        float val = logits[i] / temperature;
        if (val > max_val) max_val = val;
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        logits[i] = expf(logits[i] / temperature - max_val);
        sum += logits[i];
    }
    for (int i = 0; i < size; i++) logits[i] /= sum;

    ProbIndex* probindex = new ProbIndex[size];
    for (int i = 0; i < size; i++) { probindex[i].prob = logits[i]; probindex[i].index = i; }
    std::sort(probindex, probindex + size, compareProbIndex);

    float cumulative_prob = 0.0f; int last_idx = 0;
    for (int i = 0; i < size; i++) {
        cumulative_prob += probindex[i].prob;
        last_idx = i;
        if (cumulative_prob >= topp) break;
    }

    float r = (float)rand() / (float)RAND_MAX * cumulative_prob;
    float cdf = 0.0f; int selected_token = probindex[last_idx].index;

    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) { selected_token = probindex[i].index; break; }
    }
    delete[] probindex;
    return selected_token;
}
