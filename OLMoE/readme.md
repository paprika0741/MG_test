1. Download Hugging Face Weights
```
from huggingface_hub import snapshot_download
SAVED_DIR = "/home/ec2-user/models/OLMoE-1B-7B-0125-Instruct" # Specify the saved directory
# Download HF checkpoints
snapshot_download(repo_id="allenai/OLMoE-1B-7B-0125-Instruct", ignore_patterns=["*.pt"], local_dir=SAVED_DIR, local_dir_use_symlinks=False)
```
2. Implement Custom Loader
Implement `tools/checkpoint/loader_olmoe_hf.py` based on `tools/checkpoint/loader_mixtral_hf.py`.

+ Model structure differences:The OLMoE Hugging Face model stores expert layers and router weights under different module paths compared to Mixtral. For example: For example: Router weights are found at model.layers.N.mlp.gate.weight
    

3. Convert hf weights to mcore weight
