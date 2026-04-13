import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import math
from tqdm import tqdm
import pandas as pd

# ==========================================
# 1. 设置模型路径
# ==========================================
MODEL_PATH = "./Llama-1B-local"

print("🚀 加载 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# ==========================================
# 2. 加载 WikiText-2 测试集 (本地 parquet 文件)
# ==========================================
parquet_file = "test-00000-of-00001.parquet"
if os.path.exists(parquet_file):
    print(f"📖 从本地 parquet 文件加载: {parquet_file}")
    df = pd.read_parquet(parquet_file)
    raw_text = "\n\n".join(df["text"].tolist())
else:
    # 备用方案：如果有手动下载的纯文本
    txt_file = "wikitext-2-test.txt"
    if os.path.exists(txt_file):
        print(f"📖 从本地文本文件加载: {txt_file}")
        with open(txt_file, "r", encoding="utf-8") as f:
            raw_text = f.read()
    else:
        print("❌ 找不到测试数据！请下载 parquet 文件放到当前目录。")
        print("   下载地址: https://hf-mirror.com/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/test-00000-of-00001.parquet")
        exit(1)

print("🚀 Tokenize 中...")
encodings = tokenizer(raw_text, return_tensors="pt")
seq_len = encodings.input_ids.size(1)
print(f"✅ 总 Token 数量: {seq_len}")

# ==========================================
# 3. INT8 Group-64 伪量化
# ==========================================
def apply_fake_quantization_w8a32(model, group_size=64):
    print(f"🔪 应用 INT8 Group-{group_size} 伪量化...")
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                if "lm_head" in name:
                    continue
                weight = module.weight.data
                out_features, in_features = weight.shape
                num_groups = in_features // group_size
                w_grouped = weight.view(out_features, num_groups, group_size)
                w_max = w_grouped.abs().max(dim=-1, keepdim=True)[0]
                scale = w_max / 127.0
                scale.clamp_(min=1e-7)
                w_int8 = torch.round(w_grouped / scale).clamp(-128, 127)
                w_dequant = w_int8 * scale
                module.weight.data = w_dequant.view(out_features, in_features)
    print("✅ 伪量化完成")

# ==========================================
# 4. PPL 计算
# ==========================================
def evaluate_ppl(model, encodings, stride=512, max_length=2048):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride), desc="Evaluating PPL"):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss
        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()

# ==========================================
# 5. 运行
# ==========================================
if __name__ == "__main__":
    print("\n" + "="*50)
    print("实验 A: FP16 Baseline")
    model_baseline = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16)
    ppl_baseline = evaluate_ppl(model_baseline, encodings)
    print(f"📊 Baseline PPL: {ppl_baseline:.4f}")
    del model_baseline
    torch.cuda.empty_cache()

    print("\n" + "="*50)
    print("实验 B: INT8 Group-64 (保留 lm_head FP16)")
    model_quantized = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16)
    apply_fake_quantization_w8a32(model_quantized, group_size=64)
    ppl_quantized = evaluate_ppl(model_quantized, encodings)
    print(f"📊 INT8 PPL: {ppl_quantized:.4f}")

    print("\n" + "="*50)
    print(f"🏆 精度损失: {ppl_quantized - ppl_baseline:.4f} PPL points")
    print("="*50)
