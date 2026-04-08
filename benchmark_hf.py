import os
# 🚀 强行注入国内 HuggingFace 镜像加速下载，绕过网络阻断
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
import warnings
warnings.filterwarnings('ignore')

print("🚀 初始化 HuggingFace PyTorch Baseline...")

# 🚀 替换为 Unsloth 提供的无权限限制 (Un-gated) 等价模型
model_path = "unsloth/Llama-3.2-1B-Instruct" 

# 后面的代码完全保持不变...
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float16, device_map="cuda")
# ...

prompts = [
    "Explain the theory of relativity in simple terms.",
    "Write a C++ program to reverse a linked list.",
    "Translate the following sentence to French: 'The weather is nice today.'",
    "Summarize the main differences between CPU and GPU architectures."
]

print("\n==================================================")
print("🔥 PyTorch (HF) Benchmark Suite Initiated")
print("==================================================")

for i, prompt in enumerate(prompts):
    print(f"\n[{i+1}/4] Prompt: {prompt}")
    
    # 1. 拆解动作一：只套用 Llama 3 的对话模板，生成纯文本的 String
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # 2. 拆解动作二：使用标准的 tokenizer 编码，并送入 GPU
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # 3. 关键修复：使用 **inputs 将字典解包 (传进去 input_ids 和 attention_mask)
    generation_kwargs = dict(
        **inputs, 
        streamer=streamer, 
        max_new_tokens=128, 
        do_sample=False, 
        pad_token_id=tokenizer.eos_token_id
    )
    
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    
    # 测速开始
    start_prefill = time.perf_counter()
    thread.start()
    
    first_token_time = None
    generated_text = ""
    token_count = 0
    
    print("   [Output]: ", end="", flush=True)
    for new_text in streamer:
        if first_token_time is None:
            first_token_time = time.perf_counter()
            ttft_ms = (first_token_time - start_prefill) * 1000
        generated_text += new_text
        print(new_text, end="", flush=True)
        token_count += 1
        
    end_decode = time.perf_counter()
    print()
    
    # 排除掉 TTFT 的时间，专门计算 Decode 阶段的吞吐
    decode_time = end_decode - first_token_time
    if token_count > 1:
        tpot_ms = (decode_time * 1000) / (token_count - 1)
        throughput = (token_count - 1) / decode_time
    else:
        tpot_ms = 0
        throughput = 0
        
    print(f"   -> [Metrics] Context: {inputs['input_ids'].shape[1]} | Generated: {token_count} chunks")
    print(f"   -> ⏱️ TTFT (Prefill):   {ttft_ms:.2f} ms")
    print(f"   -> ⏱️ TPOT (Decode):    {tpot_ms:.2f} ms/tok")
    print(f"   -> 🚀 Throughput:      {throughput:.2f} tok/s")

print("\n✅ Benchmark Complete.")

peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
print(f"🔥 PyTorch 峰值专用显存占用 (Peak VRAM): {peak_memory_mb:.2f} MB")
print("⚠️ 此时请打开任务管理器 (Task Manager) -> 性能 -> GPU，记录下 Python 进程吃掉了多少专用 GPU 内存 (VRAM)！")