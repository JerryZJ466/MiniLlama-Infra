#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <vector>
#include <string>

// Tokenizer and Prompt Encoding functions
std::vector<std::string> load_vocab(int vocab_size);
std::vector<int> encode_prompt(std::string text, const std::vector<std::string>& vocab);

#endif // TOKENIZER_H