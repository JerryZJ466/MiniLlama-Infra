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
    
    // 🔪 量化拆分：INT8 权重 (q) + FP32 缩放 (s)
    int8_t* wq_q; float* wq_s;
    int8_t* wk_q; float* wk_s;
    int8_t* wv_q; float* wv_s;
    int8_t* wo_q; float* wo_s;
    
    float* rms_ffn_weight;
    
    int8_t* w1_q; float* w1_s;
    int8_t* w2_q; float* w2_s;
    int8_t* w3_q; float* w3_s;
    
    float* rms_final_weight;
    float* wcls;
};

struct TransformerWeightsGPU {
    float* token_embedding_table;
    float* rms_att_weight;
    
    int8_t* wq_q; float* wq_s;
    int8_t* wk_q; float* wk_s;
    int8_t* wv_q; float* wv_s;
    int8_t* wo_q; float* wo_s;
    
    float* rms_ffn_weight;
    
    int8_t* w1_q; float* w1_s;
    int8_t* w2_q; float* w2_s;
    int8_t* w3_q; float* w3_s;
    
    float* rms_final_weight;
    float* wcls;
};

struct RunStateGPU {
    float *x, *xb, *xb2, *hb, *hb2, *q, *k, *v, *att, *logits;
    float *key_cache, *value_cache;
};

// ==========================================
// 🚀 函数声明
// ==========================================
void rmsnorm_cuda(float* o, float* x, float* weight, int size);
void matmul(float* xout, float* x, float* w, int n, int d);
void matmul_int8(float* xout, float* x, int8_t* w_q, float* w_s, int n, int d, int group_size); // 👈 新增声明
void softmax_cuda(float* x, int size);

float* forward_cuda(int token, int pos, Config* p, TransformerWeightsGPU* w, RunStateGPU* s);
int argmax(float* logits, int size);

void alloc_gpu_weights(TransformerWeightsGPU* w_gpu, TransformerWeights* w_cpu, Config* p);
void alloc_gpu_state(RunStateGPU* s_gpu, Config* p);
void free_gpu_weights(TransformerWeightsGPU* w_gpu);
void free_gpu_state(RunStateGPU* s_gpu);
int sample_topp(float* logits, int size, float temperature, float topp);

#endif // MODEL_H