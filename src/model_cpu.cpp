#include "../include/model.h"
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <algorithm>

void rmsnorm(float* o, float* x, float* weight, int size) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) ss += x[j] * x[j];
    ss /= size;
    ss += 1e-5f; 
    ss = 1.0f / sqrtf(ss);
    for (int j = 0; j < size; j++) o[j] = weight[j] * (ss * x[j]);
}

void matmul(float* xout, float* x, float* w, int n, int d) {
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) val += w[i * n + j] * x[j];
        xout[i] = val;
    }
}

void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) x[i] /= sum;
}

float* forward(int token, int pos, Config* p, TransformerWeights* w, RunState* s) {
    int dim = p->dim, hidden_dim = p->hidden_dim, kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; 
    int head_size = dim / p->n_heads;

    memcpy(s->x, w->token_embedding_table + token * dim, dim * sizeof(float));

    for (int l = 0; l < p->n_layers; l++) {
        rmsnorm(s->xb, s->x, w->rms_att_weight + l * dim, dim);
        
        matmul(s->q, s->xb, w->wq + l * dim * dim, dim, dim);
        matmul(s->k, s->xb, w->wk + l * dim * kv_dim, dim, kv_dim);
        matmul(s->v, s->xb, w->wv + l * dim * kv_dim, dim, kv_dim);

        for (int i = 0; i < dim; i+=2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val), fci = sinf(val);
            float q0 = s->q[i], q1 = s->q[i+1];
            s->q[i] = q0 * fcr - q1 * fci; s->q[i+1] = q0 * fci + q1 * fcr;
            if (i < kv_dim) {
                float k0 = s->k[i], k1 = s->k[i+1];
                s->k[i] = k0 * fcr - k1 * fci; s->k[i+1] = k0 * fci + k1 * fcr;
            }
        }

        int loff = l * p->seq_len * kv_dim;
        memcpy(s->key_cache + loff + pos * kv_dim, s->k, kv_dim * sizeof(float));
        memcpy(s->value_cache + loff + pos * kv_dim, s->v, kv_dim * sizeof(float));

        for (int h = 0; h < p->n_heads; h++) {
            float* q = s->q + h * head_size;
            float* att = s->att + h * p->seq_len;
            for (int t = 0; t <= pos; t++) {
                float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) score += q[i] * k[i];
                score /= sqrtf(head_size);
                att[t] = score;
            }
            softmax(att, pos + 1);
            float* xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float a = att[t];
                for (int i = 0; i < head_size; i++) xb[i] += a * v[i];
            }
        }
        matmul(s->xb2, s->xb, w->wo + l * dim * dim, dim, dim);
        for (int i = 0; i < dim; i++) s->x[i] += s->xb2[i]; 

        rmsnorm(s->xb, s->x, w->rms_ffn_weight + l * dim, dim);
        matmul(s->hb, s->xb, w->w1 + l * dim * hidden_dim, dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w3 + l * dim * hidden_dim, dim, hidden_dim);
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            val *= (1.0f / (1.0f + expf(-val))); 
            val *= s->hb2[i];
            s->hb[i] = val;
        }
        matmul(s->xb, s->hb, w->w2 + l * dim * hidden_dim, hidden_dim, dim);
        for (int i = 0; i < dim; i++) s->x[i] += s->xb[i]; 
    }

    rmsnorm(s->x, s->x, w->rms_final_weight, dim);
    matmul(s->logits, s->x, w->wcls, dim, p->vocab_size);
    return s->logits;
}

int argmax(float* logits, int size) {
    int max_i = 0; float max_val = logits[0];
    for (int i = 1; i < size; i++) {
        if (logits[i] > max_val) { max_val = logits[i]; max_i = i; }
    }
    return max_i;
}

// 辅助结构体，用来给词和概率绑定排序
struct ProbIndex {
    float prob;
    int index;
};

// 排序规则：概率大的排前面
bool compareProbIndex(const ProbIndex& a, const ProbIndex& b) {
    return a.prob > b.prob;
}

int sample_topp(float* logits, int size, float temperature, float topp) {
    if (temperature < 1e-6f) {
        int max_i = 0; float max_val = logits[0];
        for (int i = 1; i < size; i++) {
            if (logits[i] > max_val) { max_val = logits[i]; max_i = i; }
        }
        return max_i;
    }

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
    for (int i = 0; i < size; i++) {
        probindex[i].prob = logits[i];
        probindex[i].index = i;
    }

    std::sort(probindex, probindex + size, compareProbIndex);

    float cumulative_prob = 0.0f;
    int last_idx = 0;
    for (int i = 0; i < size; i++) {
        cumulative_prob += probindex[i].prob;
        last_idx = i;
        if (cumulative_prob >= topp) break; 
    }

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

    delete[] probindex; 
    return selected_token;
}