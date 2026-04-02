#ifndef CONFIG_H
#define CONFIG_H

// Model parameter configuration
struct Config {
    int dim; int hidden_dim; int n_layers; int n_heads;
    int n_kv_heads; int vocab_size; int seq_len;
};

// Model weight pointers
struct TransformerWeights {
    float* token_embedding_table;
    float* rms_att_weight; float* wq; float* wk; float* wv; float* wo;
    float* rms_ffn_weight; float* w1; float* w2; float* w3;
    float* rms_final_weight; float* wcls;
};

// Runtime state and activation buffers
struct RunState {
    float *x, *xb, *xb2, *hb, *hb2, *q, *k, *v, *att, *logits;
    float *key_cache, *value_cache;
};

#endif // CONFIG_H