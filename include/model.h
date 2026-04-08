#ifndef MODEL_H
#define MODEL_H

#include "config.h"
#include <stdint.h>

// ==========================================
// 🚀 GPU VRAM Memory Pool Structures
// ==========================================
struct TransformerWeights {
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

struct RunStateGPU {
    float *x;      
    float *xb;     
    float *xb2;    
    float *hb;     
    float *hb2;    
    float *q;      
    float *k;      
    float *v;      
    float *att;    
    float *logits; 
    
    float *key_cache;
    float *value_cache;
};

// ==========================================
// 🚀 函数声明
// ==========================================
void rmsnorm_cuda(float* o, float* x, float* weight, int size);
void matmul(float* xout, float* x, float* w, int n, int d);
void softmax_cuda(float* x, int size);

float* forward_cuda(int token, int pos, Config* p, TransformerWeightsGPU* w, RunStateGPU* s);
int argmax(float* logits, int size);

void alloc_gpu_weights(TransformerWeightsGPU* w_gpu, TransformerWeights* w_cpu, Config* p);
void alloc_gpu_state(RunStateGPU* s_gpu, Config* p);
void free_gpu_weights(TransformerWeightsGPU* w_gpu);
void free_gpu_state(RunStateGPU* s_gpu);
int sample_topp(float* logits, int size, float temperature, float topp);

#endif // MODEL_H