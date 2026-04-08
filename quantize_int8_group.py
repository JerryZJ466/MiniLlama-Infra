import numpy as np
import struct


print("🚀 升级版压缩机：行级 (Row-wise) INT8 量化...")

input_path = "llama3_2_1B.bin"
output_path = "llama3_2_1B_q8_row.bin" # 新名字！

# 读取 Config
with open(input_path, "rb") as f:
    config_bytes = f.read(28)
    dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len = struct.unpack("7i", config_bytes)

kv_dim = (dim * n_kv_heads) // n_heads

sizes = {
    "token_embedding_table": vocab_size * dim,
    "rms_att_weight": n_layers * dim,
    "wq": n_layers * dim * dim, "wk": n_layers * dim * kv_dim, "wv": n_layers * dim * kv_dim, "wo": n_layers * dim * dim,
    "rms_ffn_weight": n_layers * dim,
    "w1": n_layers * dim * hidden_dim, "w2": n_layers * hidden_dim * dim, "w3": n_layers * dim * hidden_dim,
    "rms_final_weight": dim,
    "wcls": vocab_size * dim
}

# 每一行包含多少列？(用于 Reshape)
n_cols = {"wq": dim, "wk": dim, "wv": dim, "wo": dim, "w1": dim, "w2": hidden_dim, "w3": dim}
quantize_keys = ["wq", "wk", "wv", "wo", "w1", "w2", "w3"]

weights_f32 = np.fromfile(input_path, dtype=np.float32, offset=28)

print("🚀 升级版压缩机：分组 (Group-wise) INT8 量化...")

input_path = "llama3_2_1B.bin"
output_path = "llama3_2_1B_q8_group.bin"
group_size = 64 # 👑 核心参数：每 64 个元素分配一个 Scale

# ... (读取 Config 的代码保持不变) ...

with open(output_path, "wb") as f_out:
    f_out.write(config_bytes)
    offset = 0
    for name, size in sizes.items():
        tensor = weights_f32[offset : offset + size]
        offset += size
        
        if name in quantize_keys:
            cols = n_cols[name]
            rows = size // cols
            groups_per_row = cols // group_size
            
            # 🔪 分组量化核心逻辑
            # 将 [rows, cols] 变形为 [rows, groups_per_row, group_size]
            matrix = tensor.reshape(rows, groups_per_row, group_size)
            
            # 在最后一个维度 (group_size) 上求绝对值最大值
            amax = np.max(np.abs(matrix), axis=2, keepdims=True)
            amax[amax == 0] = 1e-9 # 防除零
            scale = amax / 127.0
            
            # 量化并裁剪
            matrix_q8 = np.round(matrix / scale)
            matrix_q8 = np.clip(matrix_q8, -127, 127).astype(np.int8)
            
            # 扁平化写出
            scale_f32 = scale.flatten().astype(np.float32)
            
            # 注意：现在的 Scale 数量是 rows * groups_per_row
            f_out.write(scale_f32.tobytes())
            f_out.write(matrix_q8.tobytes())
            print(f"  [Group-wise] {name:15} -> 分配 {rows * groups_per_row} 个 Scale")
        else:
            f_out.write(tensor.tobytes())

print("✅ Group-wise 量化完成！")