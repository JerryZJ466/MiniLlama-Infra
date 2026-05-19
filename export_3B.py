import torch
import struct
from transformers import AutoModelForCausalLM

model_id = "unsloth/Llama-3.2-3B-Instruct"
print(f"[Info] Loading {model_id} ...")

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
config = model.config

dim        = config.hidden_size
hidden_dim = config.intermediate_size
n_layers   = config.num_hidden_layers
n_heads    = config.num_attention_heads
n_kv_heads = config.num_key_value_heads
vocab_size = config.vocab_size
seq_len    = 4096  # cap to 4096 — 3B原始131072会导致KV cache OOM

print(f"[Info] dim={dim}, hidden={hidden_dim}, layers={n_layers}, heads={n_heads}, kv_heads={n_kv_heads}, vocab={vocab_size}")

output_file = "llama3_2_3B.bin"
state_dict  = model.state_dict()

def serialize_tensor(f, tensor):
    d = tensor.detach().cpu().view(-1).numpy().astype('float32')
    f.write(d.tobytes())

print("[Info] Exporting weights in C++ layout order...")

with open(output_file, 'wb') as f:
    header = struct.pack('iiiiiii', dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len)
    f.write(header)

    serialize_tensor(f, state_dict['model.embed_tokens.weight'])

    for i in range(n_layers): serialize_tensor(f, state_dict[f'model.layers.{i}.input_layernorm.weight'])
    for i in range(n_layers): serialize_tensor(f, state_dict[f'model.layers.{i}.self_attn.q_proj.weight'])
    for i in range(n_layers): serialize_tensor(f, state_dict[f'model.layers.{i}.self_attn.k_proj.weight'])
    for i in range(n_layers): serialize_tensor(f, state_dict[f'model.layers.{i}.self_attn.v_proj.weight'])
    for i in range(n_layers): serialize_tensor(f, state_dict[f'model.layers.{i}.self_attn.o_proj.weight'])

    for i in range(n_layers): serialize_tensor(f, state_dict[f'model.layers.{i}.post_attention_layernorm.weight'])

    for i in range(n_layers): serialize_tensor(f, state_dict[f'model.layers.{i}.mlp.gate_proj.weight'])
    for i in range(n_layers): serialize_tensor(f, state_dict[f'model.layers.{i}.mlp.down_proj.weight'])
    for i in range(n_layers): serialize_tensor(f, state_dict[f'model.layers.{i}.mlp.up_proj.weight'])

    serialize_tensor(f, state_dict['model.norm.weight'])
    serialize_tensor(f, state_dict['lm_head.weight'])

print(f"[Done] {output_file} exported.")
