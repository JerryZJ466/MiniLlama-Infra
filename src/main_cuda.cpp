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
#include <iomanip>

// ====================================================================
// 🚀 核心 Benchmark 模块：严格拆分 Prefill 与 Decode
// ====================================================================
void run_benchmark(Config* config, TransformerWeightsGPU* w_gpu, RunStateGPU* s_gpu, const std::vector<std::string>& vocab, float* cpu_logits) {
    std::cout << "\n==================================================" << std::endl;
    std::cout << "🔥 MiniLlama-Infra Benchmark Suite Initiated" << std::endl;
    std::cout << "==================================================\n" << std::endl;
    
// 1. 准备 4 条 Benchmark Prompt 的英文字符串（仅用于控制台打印，方便人看）
    std::vector<std::string> prompts = {
        "Explain the theory of relativity in simple terms.",
        "Write a C++ program to reverse a linked list.",
        "Translate the following sentence to French: 'The weather is nice today.'",
        "Summarize the main differences between CPU and GPU architectures."
    };

// 2. 预先 Tokenize 好的 ID 数组（真正喂给 GPU 引擎的数据，绕过 Tokenizer Bug）
    std::vector<std::vector<int>> pre_tokenized_prompts = {
        {13854, 279, 6634, 315, 29013, 304, 3449, 3878, 13},
        {8144, 264, 334, 1146, 2068, 311, 10134, 264, 10815, 1160, 13},
        {28573, 279, 2768, 11914, 311, 4141, 25, 356, 791, 9282, 374, 6555, 3432, 1184, 13},
        {49363, 1143, 279, 1925, 12062, 1990, 14266, 323, 23501, 78335, 13}
    };

    size_t logits_bytes = config->vocab_size * sizeof(float);
    int target_gen_len = 128; // 固定最大生成长度，确保对比公平

    // 3. 循环边界：直接使用 pre_tokenized_prompts.size() 获取数组长度
    for (size_t p_idx = 0; p_idx < pre_tokenized_prompts.size(); p_idx++) {
    
        // 💡 打印控制台提示词时，使用 prompts[p_idx] 打印字符串！
        std::cout << "\n[" << p_idx + 1 << "/" << pre_tokenized_prompts.size() << "] Prompt: " << prompts[p_idx] << std::endl;
    
        // 💡 真正送去处理的输入，使用预制好的 Token ID 数组！
        std::vector<int> user_tokens = pre_tokenized_prompts[p_idx];

        // ==============================================================
        // 下面的代码保持不变，继续拼接 Llama 3 Instruct 模板
        // ==============================================================
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

        // ==========================================
        // ⏱️ 阶段 1: Prefill (计算 TTFT)
        // ==========================================
        auto start_prefill = std::chrono::high_resolution_clock::now();
        
        // 处理 Prompt (Context)
        for (size_t i = 1; i < prompt_tokens.size(); i++) {
            forward_cuda(next_token, pos, config, w_gpu, s_gpu);
            next_token = prompt_tokens[i];
            pos++;
        }
        
        // 获取第一个预测的 Token
        float* d_logits = forward_cuda(next_token, pos, config, w_gpu, s_gpu);
        cudaMemcpy(cpu_logits, d_logits, logits_bytes, cudaMemcpyDeviceToHost);
        // 为了跑分稳定，我们在 Benchmark 中强制使用 argmax (温度 0)
        next_token = argmax(cpu_logits, config->vocab_size); 
        
        auto end_prefill = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ttft_ms = end_prefill - start_prefill;

        // 打印第一个词
        if (next_token >= 0 && next_token < vocab.size()) {
            std::string word = vocab[next_token];
            if (word.size() >= 2 && (unsigned char)word[0] == 0xC4 && (unsigned char)word[1] == 0xA0) word = " " + word.substr(2);
            std::cout << word << std::flush;
        }
        pos++;

        // ==========================================
        // ⏱️ 阶段 2: Decode (计算 TPOT 和 Throughput)
        // ==========================================
        int generate_count = 1; // 已经生成了第一个
        auto start_decode = std::chrono::high_resolution_clock::now();
        
        while (pos < config->seq_len && generate_count < target_gen_len) {
            d_logits = forward_cuda(next_token, pos, config, w_gpu, s_gpu);
            cudaMemcpy(cpu_logits, d_logits, logits_bytes, cudaMemcpyDeviceToHost);
            
            next_token = argmax(cpu_logits, config->vocab_size);

            if (next_token == 128000 || next_token == 128001 || next_token == 128008 || next_token == 128009) {
                break; // 遇到 EOS 停止
            }

            if (next_token >= 0 && next_token < vocab.size()) {
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
        
        // ==========================================
        // 📊 打印单条学术跑分报告
        // ==========================================
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "   -> [Metrics] Context: " << prompt_tokens.size() << " | Generated: " << generate_count << " tokens" << std::endl;
        std::cout << "   -> ⏱️ TTFT (Prefill):   " << ttft_ms.count() << " ms" << std::endl;
        std::cout << "   -> ⏱️ TPOT (Decode):    " << tpot_ms << " ms/tok" << std::endl;
        std::cout << "   -> 🚀 Throughput:      " << throughput << " tok/s" << std::endl;
    }
    std::cout << "\n==================================================" << std::endl;
    std::cout << "✅ Benchmark Complete. Save these FP32 numbers!" << std::endl;
    std::cout << "==================================================\n" << std::endl;
}

// ... 保持上面的 run_benchmark 不变 ...

// ====================================================================
// 🖥️ 主函数
// ====================================================================
int main() {
    SetConsoleOutputCP(CP_UTF8);
    std::cout << "🚀 Mini-Llama INT8 Group-wise CUDA Engine Starting..." << std::endl;

    // 1. 读取量化后的模型文件
    HANDLE hFile = CreateFileA("llama3_2_1B_q8_group.bin", GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) { std::cout << "❌ Error: Cannot find llama3_2_1B_q8_group.bin" << std::endl; return 1; }
    HANDLE hMap = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    float* data = (float*)MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);

    Config config;
    memcpy(&config, data, 28); // 读取前 28 字节头部
    config.seq_len = 1024;
    config.group_size = 64;    // 👑 强行注入我们在 Python 里设置的 Group Size

    std::cout << "🧠 Model Loaded | Parameters: 1B (INT8 Group-64) | Dim: " << config.dim << std::endl;

    // 2. ⚡ 极其硬核的内存映射：解析 [Scale] + [INT8 Weight] 的内存块
    TransformerWeights weights_cpu;
    char* byte_ptr = (char*)(data + 7); // 跳过 28 字节 config
    int dim = config.dim;
    int hidden_dim = config.hidden_dim;
    int kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;
    int gs = config.group_size;

    weights_cpu.token_embedding_table = (float*)byte_ptr; byte_ptr += config.vocab_size * dim * 4;
    weights_cpu.rms_att_weight = (float*)byte_ptr; byte_ptr += config.n_layers * dim * 4;

    // 🔪 wq
    int wq_elems = config.n_layers * dim * dim;
    weights_cpu.wq_s = (float*)byte_ptr; byte_ptr += (wq_elems / gs) * 4;
    weights_cpu.wq_q = (int8_t*)byte_ptr; byte_ptr += wq_elems * 1;
    // 🔪 wk
    int wk_elems = config.n_layers * dim * kv_dim;
    weights_cpu.wk_s = (float*)byte_ptr; byte_ptr += (wk_elems / gs) * 4;
    weights_cpu.wk_q = (int8_t*)byte_ptr; byte_ptr += wk_elems * 1;
    // 🔪 wv
    int wv_elems = config.n_layers * dim * kv_dim;
    weights_cpu.wv_s = (float*)byte_ptr; byte_ptr += (wv_elems / gs) * 4;
    weights_cpu.wv_q = (int8_t*)byte_ptr; byte_ptr += wv_elems * 1;
    // 🔪 wo
    int wo_elems = config.n_layers * dim * dim;
    weights_cpu.wo_s = (float*)byte_ptr; byte_ptr += (wo_elems / gs) * 4;
    weights_cpu.wo_q = (int8_t*)byte_ptr; byte_ptr += wo_elems * 1;

    weights_cpu.rms_ffn_weight = (float*)byte_ptr; byte_ptr += config.n_layers * dim * 4;

    // 🔪 w1
    int w1_elems = config.n_layers * dim * hidden_dim;
    weights_cpu.w1_s = (float*)byte_ptr; byte_ptr += (w1_elems / gs) * 4;
    weights_cpu.w1_q = (int8_t*)byte_ptr; byte_ptr += w1_elems * 1;
    // 🔪 w2
    int w2_elems = config.n_layers * hidden_dim * dim;
    weights_cpu.w2_s = (float*)byte_ptr; byte_ptr += (w2_elems / gs) * 4;
    weights_cpu.w2_q = (int8_t*)byte_ptr; byte_ptr += w2_elems * 1;
    // 🔪 w3
    int w3_elems = config.n_layers * dim * hidden_dim;
    weights_cpu.w3_s = (float*)byte_ptr; byte_ptr += (w3_elems / gs) * 4;
    weights_cpu.w3_q = (int8_t*)byte_ptr; byte_ptr += w3_elems * 1;

    weights_cpu.rms_final_weight = (float*)byte_ptr; byte_ptr += dim * 4;
    weights_cpu.wcls = weights_cpu.token_embedding_table; 

// 3. VRAM 申请与跑分
    std::cout << "📦 Allocating Static VRAM Pool for INT8..." << std::endl;

    // 🌟 [新增] 探针 1：在申请显存前，查询 GPU 当前剩余显存
    size_t free_mem_before, total_mem;
    cudaMemGetInfo(&free_mem_before, &total_mem);

    TransformerWeightsGPU w_gpu;
    RunStateGPU s_gpu;
    alloc_gpu_weights(&w_gpu, &weights_cpu, &config);
    alloc_gpu_state(&s_gpu, &config);

    // 🌟 [新增] 探针 2：在静态显存池分配完毕后，再次查询剩余显存
    size_t free_mem_after, dummy_total;
    cudaMemGetInfo(&free_mem_after, &dummy_total);

    // 🌟 [新增] 计算精准的显存占用差值
    double allocated_vram_mb = (free_mem_before - free_mem_after) / (1024.0 * 1024.0);
    
    std::cout << "==================================================" << std::endl;
    std::cout << "🔥 MiniLlama-Infra 峰值专用显存占用 (Peak VRAM): " << allocated_vram_mb << " MB" << std::endl;
    std::cout << "==================================================" << std::endl;

    float* cpu_logits = new float[config.vocab_size];
    std::vector<std::string> vocab = load_vocab(config.vocab_size);

    run_benchmark(&config, &w_gpu, &s_gpu, vocab, cpu_logits);
    
    // 4. 释放资源
    free_gpu_weights(&w_gpu);
    free_gpu_state(&s_gpu);
    delete[] cpu_logits;
    UnmapViewOfFile(data);
    CloseHandle(hMap);
    CloseHandle(hFile);
    
    return 0;
}