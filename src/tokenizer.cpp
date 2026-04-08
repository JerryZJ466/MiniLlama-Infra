#include "../include/tokenizer.h"
#include <fstream>
#include <iostream>

std::vector<std::string> load_vocab(int vocab_size) {
    std::vector<std::string> vocab(vocab_size);
    std::ifstream file("tokenizer.bin", std::ios::binary);
    if (!file) { std::cout << "❌ Error: Cannot find tokenizer.bin" << std::endl; exit(1); }
    int max_len; file.read((char*)&max_len, sizeof(int));
    for (int i = 0; i < vocab_size; i++) {
        float score; file.read((char*)&score, sizeof(float));
        int len; file.read((char*)&len, sizeof(int));
        std::string word(len, ' ');
        file.read(&word[0], len);
        vocab[i] = word;
    }
    return vocab;
}

// 🚀 终极安全版：真正的纯 ASCII 空格匹配
std::vector<int> encode_prompt(std::string text, const std::vector<std::string>& vocab) {
    std::vector<int> tokens;
    std::string current_word = "";
    
    // 🔍 调试开关：打印输入到底被切成了什么 Token
    std::cout << "[Tokenizer Debug] Encoding: ";

    for (int i = 0; i <= text.length(); i++) {
        char c = (i < text.length()) ? text[i] : ' '; 
        
        // 遇到标点或空格进行切分
        if (c == ' ' || c == '.' || c == '?' || c == '!' || c == ',') {
            if (!current_word.empty()) {
                
                // 💡 Llama 3 Python 导出时，前置空格就是普通的 ASCII 空格 " "
                std::string search_str_with_space = " " + current_word;
                int best_id = -1;
                
                // 如果这不是整句话的第一个词，优先寻找带前置空格的 Token
                if (!tokens.empty()) {
                    for (int v = 0; v < vocab.size(); v++) {
                        if (vocab[v] == search_str_with_space) { best_id = v; break; }
                    }
                }
                
                // 找不到带空格的，或者是第一个词，就找不带空格的
                if (best_id == -1) {
                    for (int v = 0; v < vocab.size(); v++) {
                        if (vocab[v] == current_word) { best_id = v; break; }
                    }
                }
                
                if (best_id != -1) {
                    tokens.push_back(best_id);
                    // 打印匹配到的 Token 字符和 ID
                    std::cout << "[" << vocab[best_id] << "|" << best_id << "] ";
                }
                current_word = "";
            }
            
            // 处理标点符号
            if (c != ' ') {
                std::string punc(1, c);
                int punc_id = -1;
                for (int v = 0; v < vocab.size(); v++) {
                    if (vocab[v] == punc) { punc_id = v; break; }
                }
                if (punc_id != -1) {
                    tokens.push_back(punc_id);
                    std::cout << "[" << punc << "|" << punc_id << "] ";
                }
            }
        } else {
            current_word += c;
        }
    }
    std::cout << std::endl;
    return tokens;
}