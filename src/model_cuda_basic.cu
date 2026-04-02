#include "../include/model.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <algorithm>

// ====================================================================
// 🚀 1. CUDA Kernels: Executed concurrently in GPU VRAM
// ====================================================================
// W shape (d, n), x shape (n,), output xout shape (d,)
__global__ void matmul_kernel(float* xout, float* x, float* w, int n, int d) {
    // Core logic: Calculate the globally unique ID for the current thread
    // This ID determines which element of the output vector 'xout' the thread computes
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Boundary check (prevent threads from exceeding dimension d)
    if (i < d) {
        float val = 0.0f;
        // Each thread is responsible for the dot product of its own row (length n)
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

// ====================================================================
// 🖥️ 2. Host Wrapper Functions: Replaces CPU matmul
// ====================================================================
void matmul(float* xout, float* x, float* w, int n, int d) {
    // Define thread block size, typically 128, 256, or 512
    int threadsPerBlock = 256;
    
    // Calculate the number of blocks (Grid) needed to cover dimension d
    int blocksPerGrid = (d + threadsPerBlock - 1) / threadsPerBlock;
    
    // Launch Kernel using <<<Grid, Block>>> syntax
    matmul_kernel<<<blocksPerGrid, threadsPerBlock>>>(xout, x, w, n, d);
    
    // Catch potential CUDA launch errors (Good engineering practice)
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in matmul: %s\n", cudaGetErrorString(err));
    }
}

// ====================================================================
// 🚀 3. CUDA Kernel: RMSNorm (Root Mean Square Normalization)
// ====================================================================
__global__ void rmsnorm_kernel(float* o, float* x, float* weight, int size) {
    __shared__ float s_sum;
    int tid = threadIdx.x;

    if (tid == 0) s_sum = 0.0f;
    __syncthreads(); 

    if (tid < size) {
        atomicAdd(&s_sum, x[tid] * x[tid]);
    }
    __syncthreads();

    if (tid < size) {
        float ss = 1.0f / sqrtf(s_sum / size + 1e-5f);
        o[tid] = weight[tid] * (ss * x[tid]);
    }
}

// ====================================================================
// 🖥️ 4. Host Wrapper Function: RMSNorm
// ====================================================================
void rmsnorm_cuda(float* o, float* x, float* weight, int size) {
    // Since rmsnorm uses __shared__ and __syncthreads internally
    // The reduction operation must be completed within the same Block.
    // Therefore, launch exactly 1 Block with thread count equal to vector size.
    rmsnorm_kernel<<<1, size>>>(o, x, weight, size);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in rmsnorm: %s\n", cudaGetErrorString(err));
    }
}

// ====================================================================
// 🚀 5. CUDA Kernel: Softmax (Core of Attention Mechanism)
// ====================================================================
__global__ void softmax_kernel(float* x, int size) {
    __shared__ float s_max;
    __shared__ float s_sum;

    int tid = threadIdx.x;

    // 1. Find max value and initialize
    if (tid == 0) {
        float max_val = x[0];
        for (int i = 1; i < size; i++) {
            if (x[i] > max_val) max_val = x[i];
        }
        s_max = max_val;
        s_sum = 0.0f; 
    }
    __syncthreads();

    // 2. Concurrently compute exponentials and accumulate sum
    if (tid < size) {
        float val = expf(x[tid] - s_max); 
        x[tid] = val;                     
        atomicAdd(&s_sum, val);           
    }
    __syncthreads();

    // 3. Concurrently perform final normalization division
    if (tid < size) {
        x[tid] = x[tid] / s_sum;
    }
}

// ====================================================================
// 🖥️ 6. Host Wrapper Function: Softmax
// ====================================================================
void softmax_cuda(float* x, int size) {
    // Similarly, due to internal dependencies on __shared__ and __syncthreads
    // Must use 1 Block to handle all computations for the current token.
    softmax_kernel<<<1, size>>>(x, size);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in softmax: %s\n", cudaGetErrorString(err));
    }
}

// ====================================================================
// 📦 7. GPU VRAM Lifecycle Management (Memory Pool)
// ====================================================================
void alloc_gpu_weights(TransformerWeightsGPU* w_gpu, TransformerWeights* w_cpu, Config* p) {
    printf("[Info] Loading model weights into GPU VRAM...\n");
    
    // Pre-calculate memory sizes for various dimensions (in bytes)
    int dim = p->dim;
    int hidden_dim = p->hidden_dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    
    size_t s_emb = p->vocab_size * dim * sizeof(float);
    size_t s_rms_att = p->n_layers * dim * sizeof(float);
    size_t s_wq = p->n_layers * dim * dim * sizeof(float);
    size_t s_wk = p->n_layers * dim * kv_dim * sizeof(float);
    size_t s_wv = p->n_layers * dim * kv_dim * sizeof(float);
    size_t s_wo = p->n_layers * dim * dim * sizeof(float);
    size_t s_rms_ffn = p->n_layers * dim * sizeof(float);
    size_t s_w1 = p->n_layers * dim * hidden_dim * sizeof(float);
    size_t s_w2 = p->n_layers * hidden_dim * dim * sizeof(float);
    size_t s_w3 = p->n_layers * dim * hidden_dim * sizeof(float);
    size_t s_rms_final = dim * sizeof(float);

    // 1. Allocate space on GPU (cudaMalloc)
    cudaMalloc((void**)&w_gpu->token_embedding_table, s_emb);
    cudaMalloc((void**)&w_gpu->rms_att_weight, s_rms_att);
    cudaMalloc((void**)&w_gpu->wq, s_wq);
    cudaMalloc((void**)&w_gpu->wk, s_wk);
    cudaMalloc((void**)&w_gpu->wv, s_wv);
    cudaMalloc((void**)&w_gpu->wo, s_wo);
    cudaMalloc((void**)&w_gpu->rms_ffn_weight, s_rms_ffn);
    cudaMalloc((void**)&w_gpu->w1, s_w1);
    cudaMalloc((void**)&w_gpu->w2, s_w2);
    cudaMalloc((void**)&w_gpu->w3, s_w3);
    cudaMalloc((void**)&w_gpu->rms_final_weight, s_rms_final);
    
    // wcls usually shares weights with token_embedding_table, point to it to save VRAM!
    w_gpu->wcls = w_gpu->token_embedding_table;

    // 2. Blast CPU data into GPU (cudaMemcpy HostToDevice)
    cudaMemcpy(w_gpu->token_embedding_table, w_cpu->token_embedding_table, s_emb, cudaMemcpyHostToDevice);
    cudaMemcpy(w_gpu->rms_att_weight, w_cpu->rms_att_weight, s_rms_att, cudaMemcpyHostToDevice);
    cudaMemcpy(w_gpu->wq, w_cpu->wq, s_wq, cudaMemcpyHostToDevice);
    cudaMemcpy(w_gpu->wk, w_cpu->wk, s_wk, cudaMemcpyHostToDevice);
    cudaMemcpy(w_gpu->wv, w_cpu->wv, s_wv, cudaMemcpyHostToDevice);
    cudaMemcpy(w_gpu->wo, w_cpu->wo, s_wo, cudaMemcpyHostToDevice);
    cudaMemcpy(w_gpu->rms_ffn_weight, w_cpu->rms_ffn_weight, s_rms_ffn, cudaMemcpyHostToDevice);
    cudaMemcpy(w_gpu->w1, w_cpu->w1, s_w1, cudaMemcpyHostToDevice);
    cudaMemcpy(w_gpu->w2, w_cpu->w2, s_w2, cudaMemcpyHostToDevice);
    cudaMemcpy(w_gpu->w3, w_cpu->w3, s_w3, cudaMemcpyHostToDevice);
    cudaMemcpy(w_gpu->rms_final_weight, w_cpu->rms_final_weight, s_rms_final, cudaMemcpyHostToDevice);

    // Ensure all transfers are complete
    cudaDeviceSynchronize();
    printf("[Success] All weights successfully loaded into VRAM!\n");
}

void alloc_gpu_state(RunStateGPU* s_gpu, Config* p) {
    int dim = p->dim;
    int hidden_dim = p->hidden_dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;

    // Allocate intermediate variable space (Note: Only Malloc is needed, no Memcpy as they are initially empty)
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
    
    // KV Cache (The source of LLM memory, occupies the largest chunk of VRAM)
    cudaMalloc((void**)&s_gpu->key_cache, p->n_layers * p->seq_len * kv_dim * sizeof(float));
    cudaMalloc((void**)&s_gpu->value_cache, p->n_layers * p->seq_len * kv_dim * sizeof(float));
}

void free_gpu_weights(TransformerWeightsGPU* w_gpu) {
    cudaFree(w_gpu->token_embedding_table);
    cudaFree(w_gpu->rms_att_weight);
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
// 🔧 8. Supplementary Kernels: Simple Element-wise concurrent operations
// ====================================================================

// (1) Vector Addition (Residual Connection): x = x + y
__global__ void add_kernel(float* x, float* y, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        x[i] += y[i];
    }
}

// (2) SiLU Activation and Gated Multiplication: hb = hb * sigmoid(hb) * hb2
__global__ void silu_mul_kernel(float* hb, float* hb2, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float val = hb[i];
        val *= (1.0f / (1.0f + expf(-val)));
        val *= hb2[i];
        hb[i] = val;
    }
}

// (3) RoPE Rotary Positional Embedding (2D Plane Rotation)
__global__ void rope_kernel(float* q, float* k, int pos, int dim, int kv_dim, int head_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // Process pairs of numbers (even indices)
    if (i < dim && i % 2 == 0) {
        int head_dim = i % head_size;
        float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
        float val = pos * freq;
        float fcr = cosf(val), fci = sinf(val);

        // Rotate Query
        float q0 = q[i], q1 = q[i+1];
        q[i]   = q0 * fcr - q1 * fci;
        q[i+1] = q0 * fci + q1 * fcr;

        // Rotate Key
        if (i < kv_dim) {
            float k0 = k[i], k1 = k[i+1];
            k[i]   = k0 * fcr - k1 * fci;
            k[i+1] = k0 * fci + k1 * fcr;
        }
    }
}

// ====================================================================
// 👁️ 11. Ultimate Kernel: CUDA Attention
// ====================================================================

// Kernel 1: Compute Q and K dot product scores (1 Block per Head, 1 Thread per historical Token)
__global__ void attention_query_key_kernel(float* att, float* q, float* key_cache, int head_size, int kv_dim, int kv_mul, int seq_len, int loff, int pos) {
    int h = blockIdx.x;   // Current Head index
    int t = threadIdx.x;  // Currently computing correlation with the t-th historical Token

    if (t <= pos) {
        float* h_q = q + h * head_size;
        // Grouped Query Attention addressing magic
        float* h_k = key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
        
        float score = 0.0f;
        for (int i = 0; i < head_size; i++) {
            score += h_q[i] * h_k[i];
        }
        // Scale by dividing by sqrt(d)
        att[h * seq_len + t] = score / sqrtf((float)head_size);
    }
}

// Kernel 2: Multiply scores with V and sum (1 Block per Head, 1 Thread per feature dimension scalar)
__global__ void attention_value_kernel(float* xb, float* att, float* value_cache, int head_size, int kv_dim, int kv_mul, int seq_len, int loff, int pos) {
    int h = blockIdx.x;
    int i = threadIdx.x;

    if (i < head_size) {
        float val = 0.0f;
        for (int t = 0; t <= pos; t++) {
            float* h_v = value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
            float a = att[h * seq_len + t]; // Retrieve the score after softmax
            val += a * h_v[i];
        }
        xb[h * head_size + i] = val;
    }
}

// ====================================================================
// 🚀 9. Ultimate Main Loop: End-to-End CUDA Forward Propagation
// ====================================================================
// Note: All data in this function flows entirely within the GPU!
float* forward_cuda(int token, int pos, Config* p, TransformerWeightsGPU* w, RunStateGPU* s) {
    int dim = p->dim;
    int hidden_dim = p->hidden_dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int head_size = dim / p->n_heads;

    int threads = 256;
    int blocks_dim = (dim + threads - 1) / threads;
    int blocks_hidden = (hidden_dim + threads - 1) / threads;

    // 1. Move current token's Embedding to s->x in VRAM
    // Note: This is a DeviceToDevice copy, extremely fast!
    cudaMemcpy(s->x, w->token_embedding_table + token * dim, dim * sizeof(float), cudaMemcpyDeviceToDevice);

    // 2. Iterate through each Transformer Layer
    for (int l = 0; l < p->n_layers; l++) {
        // [1] RMSNorm before Attention
        rmsnorm_cuda(s->xb, s->x, w->rms_att_weight + l * dim, dim);
        
        // [2] QKV Matrix Multiplication
        matmul(s->q, s->xb, w->wq + l * dim * dim, dim, dim);
        matmul(s->k, s->xb, w->wk + l * dim * kv_dim, dim, kv_dim);
        matmul(s->v, s->xb, w->wv + l * dim * kv_dim, dim, kv_dim);

        // [3] RoPE Rotary Positional Embedding
        rope_kernel<<<blocks_dim, threads>>>(s->q, s->k, pos, dim, kv_dim, head_size);

        // [4] Store K, V into KV Cache
        int loff = l * p->seq_len * kv_dim;
        cudaMemcpy(s->key_cache + loff + pos * kv_dim, s->k, kv_dim * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(s->value_cache + loff + pos * kv_dim, s->v, kv_dim * sizeof(float), cudaMemcpyDeviceToDevice);

        // [5] Fully Awakened CUDA Attention!
        // Step A: Calculate Q and K scores
        int threads_pos = 256; 
        attention_query_key_kernel<<<p->n_heads, threads_pos>>>(s->att, s->q, s->key_cache, head_size, kv_dim, kv_mul, p->seq_len, loff, pos);

        // Step B: Must synchronize here, softmax needs deterministic data
        cudaDeviceSynchronize(); 

        // Step C: Apply Softmax probability normalization for each Head's scores
        for (int h = 0; h < p->n_heads; h++) {
            softmax_cuda(s->att + h * p->seq_len, pos + 1);
        }

        // Step D: Multiply probabilities by V to extract information
        int threads_head = head_size; 
        attention_value_kernel<<<p->n_heads, threads_head>>>(s->xb, s->att, s->value_cache, head_size, kv_dim, kv_mul, p->seq_len, loff, pos);
        
        // [6] Output Projection (wo) and Residual Connection (Add)
        matmul(s->xb2, s->xb, w->wo + l * dim * dim, dim, dim);
        add_kernel<<<blocks_dim, threads>>>(s->x, s->xb2, dim); 

        // [7] RMSNorm before FFN
        rmsnorm_cuda(s->xb, s->x, w->rms_ffn_weight + l * dim, dim);

        // [8] FFN Core Computation
        matmul(s->hb, s->xb, w->w1 + l * dim * hidden_dim, dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w3 + l * dim * hidden_dim, dim, hidden_dim);
        
        silu_mul_kernel<<<blocks_hidden, threads>>>(s->hb, s->hb2, hidden_dim);
        
        matmul(s->xb, s->hb, w->w2 + l * dim * hidden_dim, hidden_dim, dim);

        // [9] Final Residual Connection
        add_kernel<<<blocks_dim, threads>>>(s->x, s->xb, dim); 
    }

    // 3. Final layer output normalization
    rmsnorm_cuda(s->x, s->x, w->rms_final_weight, dim);

    // 4. Calculate probability scores (Logits) for each token in the vocabulary
    matmul(s->logits, s->x, w->wcls, dim, p->vocab_size);

    return s->logits; // Returns a pointer located in VRAM!
}

// ====================================================================
// 🧠 10. Helper Function: Find the Token ID with the highest probability on CPU
// ====================================================================
int argmax(float* logits, int size) {
    int max_i = 0; 
    float max_val = logits[0];
    for (int i = 1; i < size; i++) {
        if (logits[i] > max_val) { 
            max_val = logits[i]; 
            max_i = i; 
        }
    }
    return max_i;
}

// Helper struct for binding tokens and probabilities for sorting
struct ProbIndex {
    float prob;
    int index;
};

// Sorting rule: Descending order by probability
bool compareProbIndex(const ProbIndex& a, const ProbIndex& b) {
    return a.prob > b.prob;
}

// ====================================================================
// 🎲 12. Ultimate Sampler: Temperature + Top-P (Nucleus Sampling)
// ====================================================================
int sample_topp(float* logits, int size, float temperature, float topp) {
    if (temperature < 1e-6f) {
        int max_i = 0; float max_val = logits[0];
        for (int i = 1; i < size; i++) {
            if (logits[i] > max_val) { max_val = logits[i]; max_i = i; }
        }
        return max_i;
    }

    // 1. Apply temperature and Softmax
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

    // 2. Prepare Top-P sorting
    ProbIndex* probindex = new ProbIndex[size];
    for (int i = 0; i < size; i++) {
        probindex[i].prob = logits[i];
        probindex[i].index = i;
    }

    // Sort in descending order (highest probability tokens first)
    std::sort(probindex, probindex + size, compareProbIndex);

    // 3. Truncate long-tail poison (Nucleus cutoff)
    float cumulative_prob = 0.0f;
    int last_idx = 0;
    for (int i = 0; i < size; i++) {
        cumulative_prob += probindex[i].prob;
        last_idx = i;
        if (cumulative_prob >= topp) break; // Cut off when reaching Top-P threshold!
    }

    // 4. Roll the dice among the remaining elite tokens
    float r = (float)rand() / (float)RAND_MAX * cumulative_prob;
    float cdf = 0.0f;
    int selected_token = probindex[last_idx].index;

    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            selected_token = probindex[i].index;
            break;
        }
    }

    delete[] probindex; // Free memory
    return selected_token;
}