from huggingface_hub import snapshot_download
SAVED_DIR = "/home/ec2-user/models/OLMoE-1B-7B-0125-Instruct" # Specify the saved directory
# Download HF checkpoints
snapshot_download(repo_id="allenai/OLMoE-1B-7B-0125-Instruct", ignore_patterns=["*.pt"], local_dir=SAVED_DIR, local_dir_use_symlinks=False)