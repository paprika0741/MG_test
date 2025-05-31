from huggingface_hub import snapshot_download
SAVED_DIR = "/mnt/data/Qwen1.5-MoE-A2.7B-Chat" # Specify the saved directory
# Download HF checkpoints
snapshot_download(repo_id="Qwen/Qwen1.5-MoE-A2.7B-Chat", ignore_patterns=["*.pt"], local_dir=SAVED_DIR, local_dir_use_symlinks=False)