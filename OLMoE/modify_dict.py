import torch
import os
import argparse
parser = argparse.ArgumentParser(description='Modify model checkpoint file paths.')
parser.add_argument('--root_dir', type=str, required=True, help='Root directory containing model files')
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
def modify(path):
    state  = torch.load(path,map_location="cpu", weights_only=False)
    print(state.keys())
    print(state["model"].keys())
    model_state = state["model"]
    new_model_state = OrderedDict()
    for key, value in   model_state.items():
        if value == None:
            continue
        print(f"{key}: {value.shape}")
        if "experts.local_experts." in key  :
        # "decoder.layers.0.mlp.experts.local_experts.0.linear_fc1.weight", 
        # "decoder.layers.0.mlp.experts.linear_fc1.weight0", 
            if  "_extra_state" not in key:
                parts = key.split(".")
                layer_idx = parts[2]
                expert_idx = parts[6]
                fc_name = parts[7]  # linear_fc1 or linear_fc2
                param_type = parts[-1]  # weight or _extra_state     
                # 构造新 key
                if param_type == "weight":
                    new_key = f"decoder.layers.{layer_idx}.mlp.experts.{fc_name}.weight{expert_idx}"
                new_model_state[new_key] = value  
            else: 
                parts = key.split(".")
                layer_idx = parts[2]
                expert_idx = parts[6]
                fc_name = parts[7]  # linear_fc1 or linear_fc2
                param_type = parts[-1]  # weight or _extra_state     
                new_key = f"decoder.layers.{layer_idx}.mlp.experts.{fc_name}._extra_state"
                new_model_state[new_key] = value  
            print("before:")
            print(key)
            print("after")
            print(new_key)
        else:
            new_model_state[key] = value  
    state["model"] = new_model_state
    torch.save(state,path)
    return new_model_state
for i in all_paths:
    modify(i)
 