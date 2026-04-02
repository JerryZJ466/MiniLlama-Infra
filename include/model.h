#ifndef MODEL_H
#define MODEL_H

#include "config.h"

// Basic mathematical operators (Host/CPU)
void rmsnorm(float* o, float* x, float* weight, int size);
void matmul(float* xout, float* x, float* w, int n, int d);
void softmax(float* x, int size);

// Model forward pass and greedy/nucleus sampling
float* forward(int token, int pos, Config* p, TransformerWeights* w, RunState* s);
int argmax(float* logits, int size);

// ==========================================
// 🚀 GPU VRAM Memory Pool Structures
// ==========================================

// 1. Read-only model weights (Resident in VRAM, never modified)
struct TransformerWeightsGPU {
    float* token_embedding_table;
    float* rms_att_weight;
    float* wq;
    float* wk;
    float* wv;
    float* wo;
    float* rms_ffn_weight;
    float* w1;
    float* w2;
    float* w3;
    float* rms_final_weight;
    float* wcls;
};

// 2. Intermediate variables for inference (Includes KV Cache)
struct RunStateGPU {
    float *x;      // Current input
    float *xb;     // Normalized temporary variable
    float *xb2;    // Second temporary variable
    float *hb;     // FFN hidden layer
    float *hb2;    // FFN hidden layer 2
    float *q;      // Query
    float *k;      // Key
    float *v;      // Value
    float *att;    // Attention scores
    float *logits; // Final output probabilities
    
    // The lifeline of LLM context: KV Cache
    float *key_cache;
    float *value_cache;
};

// ==========================================
// 🚀 GPU VRAM Lifecycle Management
// ==========================================
void alloc_gpu_weights(TransformerWeightsGPU* w_gpu, TransformerWeights* w_cpu, Config* p);
void alloc_gpu_state(RunStateGPU* s_gpu, Config* p);
void free_gpu_weights(TransformerWeightsGPU* w_gpu);
void free_gpu_state(RunStateGPU* s_gpu);

#endif // MODEL_H