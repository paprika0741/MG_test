from huggingface_hub import snapshot_download

SAVED_DIR = "/home/download/models/Mixtral-8x7B-v0.1"  # 保存目录

# 下载模型（忽略 .pt 文件），启用断点续传
snapshot_download(
    repo_id="mistralai/Mixtral-8x7B-v0.1",
    ignore_patterns=["*.pt"],
    local_dir=SAVED_DIR,
    local_dir_use_symlinks=False,
    resume_download=True  # ← 关键参数
)
