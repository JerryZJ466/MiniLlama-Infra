import os
from huggingface_hub import snapshot_download

# 强行注入国内镜像源，绕过网络封锁
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

print("🚀 开始从开源社区 (Unsloth) 强行拉取免申请模型权重...")

# 直接调用底层 API 下载，彻底绕过命令行报错
snapshot_download(
    repo_id="unsloth/Llama-3.2-1B-Instruct",
    local_dir="./Llama-1B-local",
    local_dir_use_symlinks=False  # 在 Windows 系统上最好关掉软链接，防止后续拷贝出错
)

print("\n✅ 模型下载并解压完成！请检查 Llama-1B-local 文件夹。")