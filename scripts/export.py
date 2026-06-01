import torch
import struct
from transformers import AutoModelForCausalLM

model_id = "unsloth/Llama-3.2-1B-Instruct"
print(f"[Info] Loading {model_id} ...")

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
config = model.config

dim = config.hidden_size
hidden_dim = config.intermediate_size
n_layers = config.num_hidden_layers
n_heads = config.num_attention_heads
n_kv_heads = config.num_key_value_heads
vocab_size = config.vocab_size
max_seq_len = config.max_position_embeddings

output_file = "llama3_2_1B.bin"
state_dict = model.state_dict()

def serialize_tensor(f, tensor):
    d = tensor.detach().cpu().view(-1).numpy().astype('float32')
    f.write(d.tobytes())

print("[Info] 开始按 C++ 期望的聚类顺序导出...")

with open(output_file, 'wb') as f:
    # 1. Header
    header = struct.pack('iiiiiii', dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, max_seq_len)
    f.write(header)

    # 2. Token Embeddings
    serialize_tensor(f, state_dict['model.embed_tokens.weight'])

    # 3. 🚨 核心修复：按矩阵类型聚合所有层！
    for i in range(n_layers): serialize_tensor(f, state_dict[f'model.layers.{i}.input_layernorm.weight'])
    for i in range(n_layers): serialize_tensor(f, state_dict[f'model.layers.{i}.self_attn.q_proj.weight'])
    for i in range(n_layers): serialize_tensor(f, state_dict[f'model.layers.{i}.self_attn.k_proj.weight'])
    for i in range(n_layers): serialize_tensor(f, state_dict[f'model.layers.{i}.self_attn.v_proj.weight'])
    for i in range(n_layers): serialize_tensor(f, state_dict[f'model.layers.{i}.self_attn.o_proj.weight'])
    
    for i in range(n_layers): serialize_tensor(f, state_dict[f'model.layers.{i}.post_attention_layernorm.weight'])
    
    for i in range(n_layers): serialize_tensor(f, state_dict[f'model.layers.{i}.mlp.gate_proj.weight']) # w1
    for i in range(n_layers): serialize_tensor(f, state_dict[f'model.layers.{i}.mlp.down_proj.weight']) # w2
    for i in range(n_layers): serialize_tensor(f, state_dict[f'model.layers.{i}.mlp.up_proj.weight'])   # w3

    # 4. Final RMSNorm
    serialize_tensor(f, state_dict['model.norm.weight'])

    # 5. Classifier Head
    serialize_tensor(f, state_dict['lm_head.weight'])

print("✅ 完美的聚合版 llama3_2_1B.bin 导出成功！")