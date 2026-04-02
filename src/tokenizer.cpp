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

std::vector<int> encode_prompt(std::string text, const std::vector<std::string>& vocab) {
    std::vector<int> tokens;
    std::string current_word = "";
    text += " "; 
    
    for (char c : text) {
        if (c == ' ' || c == ',' || c == '.') {
            if (!current_word.empty()) {
                int best_id = -1;
                for (int i = 0; i < vocab.size(); i++) {
                    std::string clean_v = vocab[i];
                    if (clean_v.length() > 0 && clean_v[0] == ' ') clean_v = clean_v.substr(1);
                    if (clean_v == current_word) { best_id = i; break; }
                }
                tokens.push_back(best_id != -1 ? best_id : 266); 
                current_word = "";
            }
            if (c == ',' || c == '.') {
                for (int i = 0; i < vocab.size(); i++) {
                    if (vocab[i] == std::string(1, c)) { tokens.push_back(i); break; }
                }
            }
        } else {
            current_word += c;
        }
    }
    return tokens;
}