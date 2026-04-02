#include <iostream>
#include <vector>
#include <string>
#include <windows.h>
#include "../include/config.h"
#include "../include/model.h"
#include "../include/tokenizer.h"
#include <chrono>
#include <cstdlib>
#include <ctime>

extern int sample_topp(float* logits, int size, float temperature, float topp);

int main() {
    srand(time(NULL));
    SetConsoleOutputCP(CP_UTF8);
    std::cout << "🚀 Mini-Llama Interactive Inference Engine (CPU) Starting..." << std::endl;

    HANDLE hFile = CreateFileA("stories110M.bin", GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) { std::cout << "❌ Error: Cannot find stories110M.bin" << std::endl; return 1; }
    HANDLE hMap = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    float* data = (float*)MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);

    Config config;
    memcpy(&config, data, sizeof(Config));
    std::cout << "🧠 Model Loaded | Parameters: 110M | Dim: " << config.dim << std::endl;

    TransformerWeights weights;
    float* ptr = data + 7;
    weights.token_embedding_table = ptr; ptr += config.vocab_size * config.dim;
    weights.rms_att_weight = ptr; ptr += config.n_layers * config.dim;
    weights.wq = ptr; ptr += config.n_layers * config.dim * config.dim;
    weights.wk = ptr; ptr += config.n_layers * config.dim * config.dim;
    weights.wv = ptr; ptr += config.n_layers * config.dim * config.dim;
    weights.wo = ptr; ptr += config.n_layers * config.dim * config.dim;
    weights.rms_ffn_weight = ptr; ptr += config.n_layers * config.dim;
    weights.w1 = ptr; ptr += config.n_layers * config.dim * config.hidden_dim;
    weights.w2 = ptr; ptr += config.n_layers * config.hidden_dim * config.dim;
    weights.w3 = ptr; ptr += config.n_layers * config.dim * config.hidden_dim;
    weights.rms_final_weight = ptr; ptr += config.dim;
    weights.wcls = weights.token_embedding_table;

    RunState state;
    state.x = new float[config.dim]; state.xb = new float[config.dim]; state.xb2 = new float[config.dim];
    state.hb = new float[config.hidden_dim]; state.hb2 = new float[config.hidden_dim];
    state.q = new float[config.dim]; state.k = new float[config.dim]; state.v = new float[config.dim];
    state.att = new float[config.n_heads * config.seq_len];
    state.logits = new float[config.vocab_size];
    state.key_cache = new float[config.n_layers * config.seq_len * config.dim];
    state.value_cache = new float[config.n_layers * config.seq_len * config.dim];

    std::vector<std::string> vocab = load_vocab(config.vocab_size);

    std::cout << "\n==========================================" << std::endl;
    std::cout << "Engine ready! Enter the beginning of a story and press Enter." << std::endl;
    
    while (true) {
        std::string user_input;
        std::cout << "\n[You]: ";
        std::getline(std::cin, user_input);
        if (user_input == "exit") break;

        std::vector<int> prompt_tokens = encode_prompt(user_input, vocab);
        int pos = 0;
        int next_token = 1;

        std::cout << "[Mini-Llama (CPU)]: ";

        // Process prompt
        for (int i = 0; i < prompt_tokens.size(); i++) {
            forward(next_token, pos, &config, &weights, &state);
            next_token = prompt_tokens[i];
            pos++;
        }
        
        // ==========================================
        // ⏱️ Start timing
        // ==========================================
        auto start_time = std::chrono::high_resolution_clock::now();
        int start_pos = pos;

        while (pos < config.seq_len) {
            float* logits = forward(next_token, pos, &config, &weights, &state);
            
            // Nucleus Sampling (Temperature = 0.8, Top-P = 0.9)
            // Note: Change 0.8f to 0.0f for strict benchmark testing
            next_token = sample_topp(logits, config.vocab_size, 0.0f, 0.9f);
            
            // Break on BOS or EOS
            if (next_token == 1 || next_token == 2) break;

            std::string word = vocab[next_token];
            if (word.length() > 0 && word[0] == ' ') word[0] = ' '; 
            std::cout << word << std::flush;
            pos++;
        }
        std::cout << std::endl;
        
        // ==========================================
        // ⏱️ End timing & Calculate Metrics
        // ==========================================
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        int generated_tokens = pos - start_pos;
        double speed = generated_tokens / elapsed.count();

        std::cout << "\n\n==========================================" << std::endl;
        std::cout << " CPU Performance Report" << std::endl;
        std::cout << "Elapsed Time: " << elapsed.count() << " s" << std::endl;
        std::cout << "Generated:    " << generated_tokens << " tokens" << std::endl;
        std::cout << "Speed:        " << speed << " tok/s" << std::endl;
        std::cout << "==========================================\n" << std::endl;
    }
    
    return 0;
}