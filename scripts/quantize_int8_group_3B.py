import numpy as np
import struct

input_path  = "llama3_2_3B.bin"
output_path = "llama3_2_3B_q8_group.bin"
group_size  = 64

print(f"[Info] Reading config from {input_path} ...")
with open(input_path, "rb") as f:
    config_bytes = f.read(28)
    dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len = struct.unpack("7i", config_bytes)

kv_dim = (dim * n_kv_heads) // n_heads
print(f"[Info] dim={dim}, hidden={hidden_dim}, layers={n_layers}, kv_dim={kv_dim}, vocab={vocab_size}")

sizes = {
    "token_embedding_table": vocab_size * dim,
    "rms_att_weight":        n_layers * dim,
    "wq":  n_layers * dim * dim,
    "wk":  n_layers * kv_dim * dim,
    "wv":  n_layers * kv_dim * dim,
    "wo":  n_layers * dim * dim,
    "rms_ffn_weight":        n_layers * dim,
    "w1":  n_layers * hidden_dim * dim,
    "w2":  n_layers * dim * hidden_dim,
    "w3":  n_layers * hidden_dim * dim,
    "rms_final_weight":      dim,
    "wcls": vocab_size * dim,
}

# n_cols = input dimension of each row (quantize along input)
n_cols = {
    "wq": dim, "wk": dim, "wv": dim, "wo": dim,
    "w1": dim, "w2": hidden_dim, "w3": dim
}
quantize_keys = list(n_cols.keys())

print(f"[Info] Loading FP32 weights ...")
weights_f32 = np.fromfile(input_path, dtype=np.float32, offset=28)

print(f"[Info] Quantizing to INT8 group-{group_size} -> {output_path}")
with open(output_path, "wb") as f_out:
    f_out.write(config_bytes)
    offset = 0
    for name, size in sizes.items():
        tensor = weights_f32[offset : offset + size]
        offset += size

        if name in quantize_keys:
            cols          = n_cols[name]
            rows          = size // cols
            groups_per_row = cols // group_size

            matrix = tensor.reshape(rows, groups_per_row, group_size)
            amax   = np.max(np.abs(matrix), axis=2, keepdims=True)
            amax[amax == 0] = 1e-9
            scale  = amax / 127.0

            matrix_q8 = np.clip(np.round(matrix / scale), -127, 127).astype(np.int8)
            f_out.write(scale.flatten().astype(np.float32).tobytes())
            f_out.write(matrix_q8.tobytes())
            print(f"  [Q8] {name:20} rows={rows:6d}  scales={rows*groups_per_row}")
        else:
            f_out.write(tensor.astype(np.float32).tobytes())
            print(f"  [FP] {name:20} size={size}")

print(f"[Done] {output_path} written.")
