#ifndef _WIN32
  #include <fcntl.h>
  #include <sys/mman.h>
  #include <sys/stat.h>
#endif

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <cstring>
#ifdef _WIN32
  #include <windows.h>
#else
  #include <unistd.h>
#endif
#include <cuda_runtime.h>
#include "../include/config.h"
#include "../include/model.h"
#include "../include/tokenizer.h"

// ====================================================================
// 🚀 核心 Benchmark 模块：严格拆分 Prefill 与 Decode
// ====================================================================
void run_benchmark(Config* config, TransformerWeightsGPU* w_gpu, RunStateGPU* s_gpu, const std::vector<std::string>& vocab, float* cpu_logits) {
    std::cout << "\n==================================================" << std::endl;
    std::cout << "🔥 MiniLlama-Infra Benchmark Suite Initiated" << std::endl;
    std::cout << "==================================================\n" << std::endl;
    
    std::vector<std::string> prompts = {
        "Explain the theory of relativity in simple terms.",
        "Write a C++ program to reverse a linked list.",
        "Translate the following sentence to French: 'The weather is nice today.'",
        "Summarize the main differences between CPU and GPU architectures."
    };

    std::vector<std::vector<int>> pre_tokenized_prompts = {
        {13854, 279, 6634, 315, 29013, 304, 3449, 3878, 13},
        {8144, 264, 334, 1146, 2068, 311, 10134, 264, 10815, 1160, 13},
        {28573, 279, 2768, 11914, 311, 4141, 25, 356, 791, 9282, 374, 6555, 3432, 1184, 13},
        {49363, 1143, 279, 1925, 12062, 1990, 14266, 323, 23501, 78335, 13}
    };

    size_t logits_bytes = config->vocab_size * sizeof(float);
    int target_gen_len = 128;

    for (size_t p_idx = 0; p_idx < pre_tokenized_prompts.size(); p_idx++) {
        std::cout << "\n[" << p_idx + 1 << "/" << pre_tokenized_prompts.size() << "] Prompt: " << prompts[p_idx] << std::endl;
        std::vector<int> user_tokens = pre_tokenized_prompts[p_idx];

        std::vector<int> prompt_tokens;
        std::vector<int> system_header = {128000, 128006, 9125, 128007, 271, 3835, 374, 264, 8490, 11956, 13, 128009};
        std::vector<int> user_header = {128006, 882, 128007, 271};
        std::vector<int> footer = {128009, 128006, 78191, 128007, 271};
        
        for (int t : system_header) prompt_tokens.push_back(t);
        for (int t : user_header) prompt_tokens.push_back(t);
        for (int t : user_tokens) prompt_tokens.push_back(t);
        for (int t : footer) prompt_tokens.push_back(t);

        int pos = 0;
        int next_token = prompt_tokens[0]; 

        std::cout << "   [Output]: ";

        auto start_prefill = std::chrono::high_resolution_clock::now();
        
        for (size_t i = 1; i < prompt_tokens.size(); i++) {
            forward_cuda(next_token, pos, config, w_gpu, s_gpu);
            next_token = prompt_tokens[i];
            pos++;
        }
        
        float* d_logits = forward_cuda(next_token, pos, config, w_gpu, s_gpu);
        cudaMemcpy(cpu_logits, d_logits, logits_bytes, cudaMemcpyDeviceToHost);
        next_token = argmax(cpu_logits, config->vocab_size); 
        
        auto end_prefill = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ttft_ms = end_prefill - start_prefill;

        if (next_token >= 0 && next_token < (int)vocab.size()) {
            std::string word = vocab[next_token];
            if (word.size() >= 2 && (unsigned char)word[0] == 0xC4 && (unsigned char)word[1] == 0xA0) word = " " + word.substr(2);
            std::cout << word << std::flush;
        }
        pos++;

        int generate_count = 1;
        auto start_decode = std::chrono::high_resolution_clock::now();
        
        while (pos < config->seq_len && generate_count < target_gen_len) {
            d_logits = forward_cuda(next_token, pos, config, w_gpu, s_gpu);
            cudaMemcpy(cpu_logits, d_logits, logits_bytes, cudaMemcpyDeviceToHost);
            
            next_token = argmax(cpu_logits, config->vocab_size);

            if (next_token == 128000 || next_token == 128001 || next_token == 128008 || next_token == 128009) {
                break;
            }

            if (next_token >= 0 && next_token < (int)vocab.size()) {
                std::string word = vocab[next_token];
                if (word.size() >= 2 && (unsigned char)word[0] == 0xC4 && (unsigned char)word[1] == 0xA0) word = " " + word.substr(2);
                std::cout << word << std::flush;
            }
            pos++;
            generate_count++;
        }
        std::cout << std::endl;
        
        auto end_decode = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> decode_sec = end_decode - start_decode;
        
        double tpot_ms = (decode_sec.count() * 1000.0) / generate_count;
        double throughput = generate_count / decode_sec.count();
        
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "   -> [Metrics] Context: " << prompt_tokens.size() << " | Generated: " << generate_count << " tokens" << std::endl;
        std::cout << "   -> TTFT (Prefill):   " << ttft_ms.count() << " ms" << std::endl;
        std::cout << "   -> TPOT (Decode):    " << tpot_ms << " ms/tok" << std::endl;
        std::cout << "   -> Throughput:      " << throughput << " tok/s" << std::endl;
    }
    std::cout << "\n==================================================" << std::endl;
    std::cout << "Benchmark Complete." << std::endl;
    std::cout << "==================================================\n" << std::endl;
}

// ====================================================================
// 📊 Context Length Scaling Benchmark
// 测试不同 context 长度下的 TTFT 和 Decode 吞吐量
// ====================================================================
void run_context_scaling_benchmark(Config* config, TransformerWeightsGPU* w_gpu, RunStateGPU* s_gpu, float* cpu_logits) {
    std::cout << "\n==================================================" << std::endl;
    std::cout << "📊 Context Length Scaling Benchmark" << std::endl;
    std::cout << "==================================================\n" << std::endl;

    // 用 token ID 1000 填充不同长度的 prompt
    // 加上 system_header(12) + user_header(4) + footer(5) = 21 token 模板开销
    std::vector<int> test_content_lengths = {32, 64, 128, 256, 512};
    
    std::vector<int> system_header = {128000, 128006, 9125, 128007, 271, 3835, 374, 264, 8490, 11956, 13, 128009};
    std::vector<int> user_header = {128006, 882, 128007, 271};
    std::vector<int> footer = {128009, 128006, 78191, 128007, 271};

    size_t logits_bytes = config->vocab_size * sizeof(float);
    int target_gen_len = 64;  // 每种长度生成 64 个 token 做 decode 测试
    int warmup_runs = 1;      // 预热 1 次

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "| Context Len | TTFT (ms) | TPOT (ms/tok) | Throughput (tok/s) |" << std::endl;
    std::cout << "|-------------|-----------|---------------|---------------------|" << std::endl;

    for (int content_len : test_content_lengths) {
        // 构造 prompt: header + content_len 个填充 token + footer
        std::vector<int> prompt_tokens;
        for (int t : system_header) prompt_tokens.push_back(t);
        for (int t : user_header) prompt_tokens.push_back(t);
        for (int i = 0; i < content_len; i++) prompt_tokens.push_back(1000); // 填充 token
        for (int t : footer) prompt_tokens.push_back(t);

        int total_context = (int)prompt_tokens.size();

        // 检查不超过 seq_len
        if (total_context + target_gen_len > config->seq_len) {
            std::cout << "| " << total_context << " | SKIP (exceeds seq_len) |" << std::endl;
            continue;
        }

        // 预热：跑一次丢掉结果
        for (int w = 0; w < warmup_runs; w++) {
            int pos = 0;
            int next_token = prompt_tokens[0];
            for (size_t i = 1; i < prompt_tokens.size(); i++) {
                forward_cuda(next_token, pos, config, w_gpu, s_gpu);
                next_token = prompt_tokens[i];
                pos++;
            }
            forward_cuda(next_token, pos, config, w_gpu, s_gpu);
        }

        // 正式测量：取 3 次平均
        double total_ttft = 0.0;
        double total_tpot = 0.0;
        double total_throughput = 0.0;
        int num_runs = 3;

        for (int run = 0; run < num_runs; run++) {
            int pos = 0;
            int next_token = prompt_tokens[0];

            // ==========================================
            // Prefill 阶段
            // ==========================================
            auto start_prefill = std::chrono::high_resolution_clock::now();

            for (size_t i = 1; i < prompt_tokens.size(); i++) {
                forward_cuda(next_token, pos, config, w_gpu, s_gpu);
                next_token = prompt_tokens[i];
                pos++;
            }

            float* d_logits = forward_cuda(next_token, pos, config, w_gpu, s_gpu);
            cudaMemcpy(cpu_logits, d_logits, logits_bytes, cudaMemcpyDeviceToHost);
            next_token = argmax(cpu_logits, config->vocab_size);

            auto end_prefill = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> ttft_ms = end_prefill - start_prefill;
            pos++;

            // ==========================================
            // Decode 阶段
            // ==========================================
            int generate_count = 1;
            auto start_decode = std::chrono::high_resolution_clock::now();

            while (pos < config->seq_len && generate_count < target_gen_len) {
                d_logits = forward_cuda(next_token, pos, config, w_gpu, s_gpu);
                cudaMemcpy(cpu_logits, d_logits, logits_bytes, cudaMemcpyDeviceToHost);
                next_token = argmax(cpu_logits, config->vocab_size);

                if (next_token == 128000 || next_token == 128001 || next_token == 128008 || next_token == 128009) {
                    break;
                }
                pos++;
                generate_count++;
            }

            auto end_decode = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> decode_sec = end_decode - start_decode;

            total_ttft += ttft_ms.count();
            total_tpot += (decode_sec.count() * 1000.0) / generate_count;
            total_throughput += generate_count / decode_sec.count();
        }

        double avg_ttft = total_ttft / num_runs;
        double avg_tpot = total_tpot / num_runs;
        double avg_throughput = total_throughput / num_runs;

        std::cout << "| " << std::setw(11) << total_context
                  << " | " << std::setw(9) << avg_ttft
                  << " | " << std::setw(13) << avg_tpot
                  << " | " << std::setw(19) << avg_throughput
                  << " |" << std::endl;
    }

    std::cout << "\n==================================================" << std::endl;
    std::cout << "Context Scaling Benchmark Complete." << std::endl;
    std::cout << "==================================================\n" << std::endl;
}

// ====================================================================
// 🖥️ 主函数
// ====================================================================
int main(int argc, char* argv[]) {
    const char* model_path = (argc > 1) ? argv[1] : "llama3_2_1B_q8_group.bin";
    int seq_len_override = (argc > 2) ? atoi(argv[2]) : 1024;

#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    HANDLE hFile = CreateFileA(model_path, GENERIC_READ, FILE_SHARE_READ,
                               NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        std::cout << "Error: Cannot open " << model_path << std::endl; return 1;
    }
    HANDLE hMap = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    float* data = (float*)MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
    size_t file_size = GetFileSize(hFile, NULL);
#else
    int fd = open(model_path, O_RDONLY);
    if (fd < 0) { printf("Error: Cannot open %s\n", model_path); return 1; }
    struct stat st; fstat(fd, &st);
    size_t file_size = st.st_size;
    float* data = (float*)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) { printf("Error: mmap failed\n"); return 1; }
    close(fd);
#endif

    Config config;
    memcpy(&config, data, 28);
    config.seq_len = seq_len_override;
    config.group_size = 64;

    std::cout << "Model Loaded | Parameters: 1B (INT8 Group-64) | Dim: " << config.dim << std::endl;

    // 2. 内存映射解析权重
    TransformerWeights weights_cpu;
    char* byte_ptr = (char*)(data + 7);
    int dim = config.dim;
    int hidden_dim = config.hidden_dim;
    int kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;
    int gs = config.group_size;

    weights_cpu.token_embedding_table = (float*)byte_ptr; byte_ptr += config.vocab_size * dim * 4;
    weights_cpu.rms_att_weight = (float*)byte_ptr; byte_ptr += config.n_layers * dim * 4;

    int wq_elems = config.n_layers * dim * dim;
    weights_cpu.wq_s = (float*)byte_ptr; byte_ptr += (wq_elems / gs) * 4;
    weights_cpu.wq_q = (int8_t*)byte_ptr; byte_ptr += wq_elems * 1;
    int wk_elems = config.n_layers * dim * kv_dim;
    weights_cpu.wk_s = (float*)byte_ptr; byte_ptr += (wk_elems / gs) * 4;
    weights_cpu.wk_q = (int8_t*)byte_ptr; byte_ptr += wk_elems * 1;
    int wv_elems = config.n_layers * dim * kv_dim;
    weights_cpu.wv_s = (float*)byte_ptr; byte_ptr += (wv_elems / gs) * 4;
    weights_cpu.wv_q = (int8_t*)byte_ptr; byte_ptr += wv_elems * 1;
    int wo_elems = config.n_layers * dim * dim;
    weights_cpu.wo_s = (float*)byte_ptr; byte_ptr += (wo_elems / gs) * 4;
    weights_cpu.wo_q = (int8_t*)byte_ptr; byte_ptr += wo_elems * 1;

    weights_cpu.rms_ffn_weight = (float*)byte_ptr; byte_ptr += config.n_layers * dim * 4;

    int w1_elems = config.n_layers * dim * hidden_dim;
    weights_cpu.w1_s = (float*)byte_ptr; byte_ptr += (w1_elems / gs) * 4;
    weights_cpu.w1_q = (int8_t*)byte_ptr; byte_ptr += w1_elems * 1;
    int w2_elems = config.n_layers * hidden_dim * dim;
    weights_cpu.w2_s = (float*)byte_ptr; byte_ptr += (w2_elems / gs) * 4;
    weights_cpu.w2_q = (int8_t*)byte_ptr; byte_ptr += w2_elems * 1;
    int w3_elems = config.n_layers * dim * hidden_dim;
    weights_cpu.w3_s = (float*)byte_ptr; byte_ptr += (w3_elems / gs) * 4;
    weights_cpu.w3_q = (int8_t*)byte_ptr; byte_ptr += w3_elems * 1;

    weights_cpu.rms_final_weight = (float*)byte_ptr; byte_ptr += dim * 4;
    weights_cpu.wcls = weights_cpu.token_embedding_table; 

    // 3. VRAM 申请
    std::cout << "Allocating Static VRAM Pool for INT8..." << std::endl;

    size_t free_mem_before, total_mem;
    cudaMemGetInfo(&free_mem_before, &total_mem);

    TransformerWeightsGPU w_gpu;
    RunStateGPU s_gpu;
    alloc_gpu_weights(&w_gpu, &weights_cpu, &config);
    alloc_gpu_state(&s_gpu, &config);

    size_t free_mem_after, dummy_total;
    cudaMemGetInfo(&free_mem_after, &dummy_total);

    double allocated_vram_mb = (free_mem_before - free_mem_after) / (1024.0 * 1024.0);
    
    std::cout << "==================================================" << std::endl;
    std::cout << "Peak VRAM: " << allocated_vram_mb << " MB" << std::endl;
    std::cout << "==================================================" << std::endl;

    float* cpu_logits = new float[config.vocab_size];
    std::vector<std::string> vocab = load_vocab(config.vocab_size);

    // 4. 运行标准 Benchmark（4 条 prompt）
    run_benchmark(&config, &w_gpu, &s_gpu, vocab, cpu_logits);

    // 5. 运行 Context Length Scaling Benchmark
    run_context_scaling_benchmark(&config, &w_gpu, &s_gpu, cpu_logits);
    
    // 6. 释放资源
    free_gpu_weights(&w_gpu);
    free_gpu_state(&s_gpu);
    delete[] cpu_logits;
#ifdef _WIN32
    UnmapViewOfFile(data);
    CloseHandle(hMap);
    CloseHandle(hFile);
#else
    munmap(data, file_size);
#endif
    
    return 0;
}