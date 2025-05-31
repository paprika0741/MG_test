import torch
import os
import argparse
parser = argparse.ArgumentParser(description='Modify model checkpoint file paths.')
parser.add_argument('--root_dir', type=str, help='Root directory containing model files', default="/home/ec2-user/CodeSpace/NEW_Megatron/Megatron-LM-core_v0.12.0/mixtral/mixtral-mcore-TP1PP1EP4Layer1")
args = parser.parse_args()
root_dir = args.root_dir
# this is only for mixtral
from collections import OrderedDict
root_dir = args.root_dir
target_name = 'model_optim_rng.pt'
if not os.path.exists(root_dir):
    print(f"not exist: {root_dir}")
    exit(0)
all_paths = []
for dirpath, dirnames, filenames in os.walk(root_dir):
    if target_name in filenames:
        full_path = os.path.join(dirpath, target_name)
        all_paths.append(full_path)
for path in all_paths:
    print(path)
def check(path):
    state  = torch.load(path,map_location="cpu", weights_only=False)
    print(state.keys())
    print(state["model"].keys())
    model_state = state["model"]
    new_model_state = OrderedDict()
    for key, value in model_state.items():
        if value == None:
            continue
        print(f"{key}: {value.shape}")
        if "_extra_state" not in key:
            tensor = model_state[key]
            total_elements = tensor.numel()
            zero_count = (tensor == 0).sum().item()
            sparsity = zero_count / total_elements
            if sparsity > 0.5:
                print("  ⚠️ Warning: Sparsity is greater than 0.5!")
            else:
                print(f"  Zero count: {zero_count}")
                print(f"  Total elements: {total_elements}")
                print(f"  Sparsity: {sparsity:.4f}")


check(all_paths [0])