#include <iostream>
#include <vector>
#include <string>
#include <windows.h>
#include <cuda_runtime.h>
#include "../include/config.h"
#include "../include/model.h"
#include "../include/tokenizer.h"
#include <chrono>
#include <cstdlib>
#include <ctime>

// Declare external GPU functions from model_cuda_basic.cu
extern float* forward_cuda(int token, int pos, Config* p, TransformerWeightsGPU* w, RunStateGPU* s);
extern int sample_topp(float* logits, int size, float temperature, float topp);
extern void alloc_gpu_weights(TransformerWeightsGPU* w_gpu, TransformerWeights* w_cpu, Config* p);
extern void alloc_gpu_state(RunStateGPU* s_gpu, Config* p);
extern void free_gpu_weights(TransformerWeightsGPU* w_gpu);
extern void free_gpu_state(RunStateGPU* s_gpu);

int main() {
    srand(time(NULL));
    SetConsoleOutputCP(CP_UTF8);
    std::cout << "🚀 Mini-Llama Pure CUDA Engine Starting..." << std::endl;

    // 1. Map model weights file to host memory
    HANDLE hFile = CreateFileA("stories110M.bin", GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) { std::cout << "❌ Error: Cannot find stories110M.bin" << std::endl; return 1; }
    HANDLE hMap = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    float* data = (float*)MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);

    Config config;
    memcpy(&config, data, sizeof(Config));
    std::cout << "🧠 Model Loaded | Parameters: 110M | Dim: " << config.dim << std::endl;

    // Setup host pointers for weights
    TransformerWeights weights_cpu;
    float* ptr = data + 7;
    weights_cpu.token_embedding_table = ptr; ptr += config.vocab_size * config.dim;
    weights_cpu.rms_att_weight = ptr; ptr += config.n_layers * config.dim;
    weights_cpu.wq = ptr; ptr += config.n_layers * config.dim * config.dim;
    weights_cpu.wk = ptr; ptr += config.n_layers * config.dim * config.dim;
    weights_cpu.wv = ptr; ptr += config.n_layers * config.dim * config.dim;
    weights_cpu.wo = ptr; ptr += config.n_layers * config.dim * config.dim;
    weights_cpu.rms_ffn_weight = ptr; ptr += config.n_layers * config.dim;
    weights_cpu.w1 = ptr; ptr += config.n_layers * config.dim * config.hidden_dim;
    weights_cpu.w2 = ptr; ptr += config.n_layers * config.hidden_dim * config.dim;
    weights_cpu.w3 = ptr; ptr += config.n_layers * config.dim * config.hidden_dim;
    weights_cpu.rms_final_weight = ptr; ptr += config.dim;
    weights_cpu.wcls = weights_cpu.token_embedding_table;

    // ====================================================================
    // ⚡ VRAM Allocation
    // ====================================================================
    TransformerWeightsGPU w_gpu;
    RunStateGPU s_gpu;
    
    // Transfer weights and allocate activation buffers on GPU
    alloc_gpu_weights(&w_gpu, &weights_cpu, &config);
    alloc_gpu_state(&s_gpu, &config);

    // Host buffer to receive final logits from GPU
    float* cpu_logits = new float[config.vocab_size];
    size_t logits_bytes = config.vocab_size * sizeof(float);

    std::vector<std::string> vocab = load_vocab(config.vocab_size);

    std::cout << "\n==========================================" << std::endl;
    std::cout << "GPU Engine ready! Enter the beginning of a story and press Enter." << std::endl;
    
    while (true) {
        std::string user_input;
        std::cout << "\n[You]: ";
        std::getline(std::cin, user_input);
        if (user_input == "exit") break;

        std::vector<int> prompt_tokens = encode_prompt(user_input, vocab);
        int pos = 0;
        int next_token = 1; // BOS Token

        std::cout << "[Mini-Llama (CUDA)]: ";

        // Process prompt
        for (int i = 0; i < prompt_tokens.size(); i++) {
            forward_cuda(next_token, pos, &config, &w_gpu, &s_gpu);
            next_token = prompt_tokens[i];
            pos++;
        }
        
        // ==========================================
        // ⏱️ Start timing
        // ==========================================
        auto start_time = std::chrono::high_resolution_clock::now();
        int start_pos = pos;
        
        // Autoregressive generation loop
        while (pos < config.seq_len) {
            // 1. Forward pass entirely on GPU
            float* d_logits = forward_cuda(next_token, pos, &config, &w_gpu, &s_gpu);
            
            // 2. [Device-to-Host] Copy logits to CPU for sampling
            cudaMemcpy(cpu_logits, d_logits, logits_bytes, cudaMemcpyDeviceToHost);

            // 3. Nucleus Sampling on CPU (Temperature = 0.8, Top-P = 0.9)
            // Note: Change 0.8f to 0.0f for strict benchmark testing
            next_token = sample_topp(cpu_logits, config.vocab_size, 0.0f, 0.9f);
            
            // Break on BOS or EOS
            if (next_token == 2 || next_token == 1) break;

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
        std::cout << " CUDA Performance Report" << std::endl;
        std::cout << "Elapsed Time: " << elapsed.count() << " s" << std::endl;
        std::cout << "Generated:    " << generated_tokens << " tokens" << std::endl;
        std::cout << "Speed:        " << speed << " tok/s" << std::endl;
        std::cout << "==========================================\n" << std::endl;
    }
    
    // Cleanup VRAM
    free_gpu_weights(&w_gpu);
    free_gpu_state(&s_gpu);
    delete[] cpu_logits;
    
    return 0;
}