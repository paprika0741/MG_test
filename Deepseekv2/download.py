from huggingface_hub import snapshot_download
SAVED_DIR = "/home/download/models/DeepSeek-V2-Lite" # Specify the saved directory
# Download HF checkpoints
snapshot_download(repo_id="deepseek-ai/DeepSeek-V2-Lite", ignore_patterns=["*.pt"], local_dir=SAVED_DIR, local_dir_use_symlinks=False)