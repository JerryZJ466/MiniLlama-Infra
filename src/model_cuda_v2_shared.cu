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
// 🚀 CUDA Kernel: RMSNorm & Softmax
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

__global__ void softmax_kernel(float* x, int size) {
    __shared__ float s_max;
    __shared__ float s_sum;
    int tid = threadIdx.x;
    if (tid == 0) {
        float max_val = x[0];
        for (int i = 1; i < size; i++) if (x[i] > max_val) max_val = x[i];
        s_max = max_val;
        s_sum = 0.0f; 
    }
    __syncthreads();
    if (tid < size) {
        float val = expf(x[tid] - s_max); 
        x[tid] = val;                     
        atomicAdd(&s_sum, val);           
    }
    __syncthreads();
    if (tid < size) x[tid] = x[tid] / s_sum;
}

void softmax_cuda(float* x, int size) {
    softmax_kernel<<<1, size>>>(x, size);
}

// ====================================================================
// 📦 GPU VRAM Management (PURE FP32)
// ====================================================================
void alloc_gpu_weights(TransformerWeightsGPU* w_gpu, TransformerWeights* w_cpu, Config* p) {
    int dim = p->dim; int hidden_dim = p->hidden_dim; int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;

    // 1. 申请纯 FP32 显存 (* 4 字节)
    cudaMalloc((void**)&w_gpu->token_embedding_table, p->vocab_size * dim * 4);
    cudaMalloc((void**)&w_gpu->rms_att_weight, p->n_layers * dim * 4);
    cudaMalloc((void**)&w_gpu->wq, p->n_layers * dim * dim * 4);
    cudaMalloc((void**)&w_gpu->wk, p->n_layers * dim * kv_dim * 4);
    cudaMalloc((void**)&w_gpu->wv, p->n_layers * dim * kv_dim * 4);
    cudaMalloc((void**)&w_gpu->wo, p->n_layers * dim * dim * 4);
    cudaMalloc((void**)&w_gpu->rms_ffn_weight, p->n_layers * dim * 4);
    cudaMalloc((void**)&w_gpu->w1, p->n_layers * dim * hidden_dim * 4);
    cudaMalloc((void**)&w_gpu->w2, p->n_layers * hidden_dim * dim * 4);
    cudaMalloc((void**)&w_gpu->w3, p->n_layers * dim * hidden_dim * 4);
    cudaMalloc((void**)&w_gpu->rms_final_weight, dim * 4);
    
    // 2. 拷贝 FP32 权重到显卡
    cudaMemcpy(w_gpu->token_embedding_table, w_cpu->token_embedding_table, p->vocab_size * dim * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(w_gpu->rms_att_weight, w_cpu->rms_att_weight, p->n_layers * dim * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(w_gpu->wq, w_cpu->wq, p->n_layers * dim * dim * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(w_gpu->wk, w_cpu->wk, p->n_layers * dim * kv_dim * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(w_gpu->wv, w_cpu->wv, p->n_layers * dim * kv_dim * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(w_gpu->wo, w_cpu->wo, p->n_layers * dim * dim * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(w_gpu->rms_ffn_weight, w_cpu->rms_ffn_weight, p->n_layers * dim * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(w_gpu->w1, w_cpu->w1, p->n_layers * dim * hidden_dim * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(w_gpu->w2, w_cpu->w2, p->n_layers * hidden_dim * dim * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(w_gpu->w3, w_cpu->w3, p->n_layers * dim * hidden_dim * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(w_gpu->rms_final_weight, w_cpu->rms_final_weight, dim * 4, cudaMemcpyHostToDevice);
    
    // 🚀 复用 wcls 指针
    w_gpu->wcls = w_gpu->token_embedding_table;
    cudaDeviceSynchronize();
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
}

void free_gpu_weights(TransformerWeightsGPU* w_gpu) {
    cudaFree(w_gpu->token_embedding_table); cudaFree(w_gpu->rms_att_weight);
    cudaFree(w_gpu->wq); cudaFree(w_gpu->wk); cudaFree(w_gpu->wv); cudaFree(w_gpu->wo);
    cudaFree(w_gpu->rms_ffn_weight);
    cudaFree(w_gpu->w1); cudaFree(w_gpu->w2); cudaFree(w_gpu->w3);
    cudaFree(w_gpu->rms_final_weight);
}

void free_gpu_state(RunStateGPU* s_gpu) {
    cudaFree(s_gpu->x); cudaFree(s_gpu->xb); cudaFree(s_gpu->xb2);
    cudaFree(s_gpu->hb); cudaFree(s_gpu->hb2);
    cudaFree(s_gpu->q); cudaFree(s_gpu->k); cudaFree(s_gpu->v);
    cudaFree(s_gpu->att); cudaFree(s_gpu->logits);
    cudaFree(s_gpu->key_cache); cudaFree(s_gpu->value_cache);
}

// ====================================================================
// 🔧 Supplementary Kernels (Add, Silu, RoPE)
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

// 🚀 Llama 3.2 专属分段 RoPE 频率计算器
__device__ float get_llama3_scaled_freq(int head_dim_idx, int head_size) {
    float freq = 1.0f / powf(500000.0f, (head_dim_idx * 2) / (float)head_size);
    float wavelength = 2.0f * 3.141592653589793f / freq;
    
    // Llama 3.2 的缩放常数
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

// 🚀 终极版 RoPE Kernel (兼容 HuggingFace 前后半区排布)
__global__ void rope_kernel(float* q, float* k, int pos, int dim, int kv_dim, int head_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 旋转 Query
    if (i < dim) {
        int head_dim_idx = i % head_size;
        // HuggingFace 格式：处理前半区并与后半区匹配
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
    
    // 旋转 Key
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

__global__ void attention_query_key_kernel(float* att, float* q, float* key_cache, int head_size, int kv_dim, int kv_mul, int seq_len, int loff, int pos) {
    int h = blockIdx.x; 
    for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
        float* h_q = q + h * head_size;
        float* h_k = key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
        float score = 0.0f;
        for (int i = 0; i < head_size; i++) score += h_q[i] * h_k[i];
        att[h * seq_len + t] = score / sqrtf((float)head_size);
    }
}

__global__ void attention_value_kernel(float* xb, float* att, float* value_cache, int head_size, int kv_dim, int kv_mul, int seq_len, int loff, int pos) {
    int h = blockIdx.x; int i = threadIdx.x;
    if (i < head_size) {
        float val = 0.0f;
        for (int t = 0; t <= pos; t++) {
            float* h_v = value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
            float a = att[h * seq_len + t]; 
            val += a * h_v[i];
        }
        xb[h * head_size + i] = val;
    }
}

// ====================================================================
// 🚀 Forward Pass (FP32)
// ====================================================================
float* forward_cuda(int token, int pos, Config* p, TransformerWeightsGPU* w, RunStateGPU* s) {
    int dim = p->dim; int hidden_dim = p->hidden_dim; int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; int head_size = dim / p->n_heads;
    int threads = 256;
    int blocks_dim = (dim + threads - 1) / threads;
    int blocks_hidden = (hidden_dim + threads - 1) / threads;

    cudaMemcpy(s->x, w->token_embedding_table + token * dim, dim * sizeof(float), cudaMemcpyDeviceToDevice);

    for (int l = 0; l < p->n_layers; l++) {
        rmsnorm_cuda(s->xb, s->x, w->rms_att_weight + l * dim, dim);
        
        matmul(s->q, s->xb, w->wq + l * dim * dim, dim, dim);
        matmul(s->k, s->xb, w->wk + l * dim * kv_dim, dim, kv_dim);
        matmul(s->v, s->xb, w->wv + l * dim * kv_dim, dim, kv_dim);

        rope_kernel<<<blocks_dim, threads>>>(s->q, s->k, pos, dim, kv_dim, head_size);

        int loff = l * p->seq_len * kv_dim;
        cudaMemcpy(s->key_cache + loff + pos * kv_dim, s->k, kv_dim * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(s->value_cache + loff + pos * kv_dim, s->v, kv_dim * sizeof(float), cudaMemcpyDeviceToDevice);

        attention_query_key_kernel<<<p->n_heads, 256>>>(s->att, s->q, s->key_cache, head_size, kv_dim, kv_mul, p->seq_len, loff, pos);
        cudaDeviceSynchronize(); 
        for (int h = 0; h < p->n_heads; h++) softmax_cuda(s->att + h * p->seq_len, pos + 1);
        attention_value_kernel<<<p->n_heads, head_size>>>(s->xb, s->att, s->value_cache, head_size, kv_dim, kv_mul, p->seq_len, loff, pos);
        
        matmul(s->xb2, s->xb, w->wo + l * dim * dim, dim, dim);
        add_kernel<<<blocks_dim, threads>>>(s->x, s->xb2, dim);

        rmsnorm_cuda(s->xb, s->x, w->rms_ffn_weight + l * dim, dim);

        matmul(s->hb, s->xb, w->w1 + l * dim * hidden_dim, dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w3 + l * dim * hidden_dim, dim, hidden_dim);
        silu_mul_kernel<<<blocks_hidden, threads>>>(s->hb, s->hb2, hidden_dim);
        matmul(s->xb, s->hb, w->w2 + l * dim * hidden_dim, hidden_dim, dim);

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