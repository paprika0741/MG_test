import torch
import os
import argparse
import re
from transformers import AutoModelForCausalLM
from transformers import AutoConfig


def get_param_name(model, target_param):
    for name, param in model.named_parameters():
        if param is target_param:
            return name
    return None  # 如果找不到，返回 None
def check_hf(path):
    mismatch_leys = []
    checked_keys = set()
    match = re.search(r'TP(\d+)PP(\d+)EP(\d+)', path)
    if match:
        tp = int(match.group(1))
        pp = int(match.group(2))
        ep = int(match.group(3))
        print(f"TP: {tp}, PP: {pp}, EP: {ep}")
    else:
        raise ValueError("Format not matched for TP/PP/EP")

        
    match = re.search(r'mp_rank_\d+_(\d+)', path)
    if match:
        ep_rank = int(match.group(1)) 
        print(f"ep_rank: {ep_rank}")
    else:
        raise ValueError("EP rank not found in path")
    # Load Megatron weight dictionary
    mg_model_state = torch.load(path,map_location="cpu", weights_only=False)["model"]
    # Compute expert partitioning info
    world_size = ep  # 总共有几个 expert ranks / GPUs
    expert_indices = list(range(config.num_experts))  # 所有 expert 的 index
    # 每个 rank 应该负责的 expert 数量
    num_local_experts = int(config.num_experts // world_size)
    # 当前 rank 负责的 expert index 范围
    start_idx = ep_rank * num_local_experts
    end_idx = start_idx + num_local_experts
    # 当前 rank 实际负责的 expert indices
    local_expert_indices = expert_indices[start_idx:end_idx]
    # 打印调试信息
    print(f"Total experts: {config.num_experts}")
    print(f"World size (EP): {world_size}")
    print(f"EP rank: {ep_rank}")
    print(f"Experts per rank: {num_local_experts}")
    print(f"Local expert indices: {local_expert_indices}")
    print("====================HF weight=====================")
    for name, param in hf_model.named_parameters():
        print(f"{name}: {param.shape}")
    print("====================MG weight=====================")
    for key, value in mg_model_state.items():
        print(key, value.shape)
        if "_extra_state" in key:
            print("pass")
            continue
        matched = False
        if "embedding" in key:
            print(torch.equal(mg_model_state[key], hf_model.model.embed_tokens.weight   ))
        if "final_layernorm" in key:
            print(torch.equal(mg_model_state[key], hf_model.model.norm.weight   ))
        if "output_layer"  in key:
            print(torch.equal(mg_model_state[key], hf_model.lm_head.weight  ))
        if "layers" in key:
            parts = key.split(".")
            layer_idx = int(parts[2])
            hf_layer = hf_model.model.layers[ layer_idx ]
            if "self_attention" in key  :
                hf_attn = hf_layer.self_attn
                if "linear_proj" in key:
                    print(torch.equal(mg_model_state[key], hf_attn.o_proj.weight    ))
                if "linear_qkv" in key:
                    if "linear_qkv.weight" in key:
                        num_query_groups=16
                        dim = 128
                        num_querys_per_group = 1
                        num_heads = 16
                        attn_weight = torch.cat([
                        hf_attn.q_proj.weight.reshape((num_query_groups, num_querys_per_group*dim, -1)),
                        hf_attn.k_proj.weight.reshape((num_query_groups, dim, -1)),
                        hf_attn.v_proj.weight.reshape((num_query_groups, dim, -1)),
                        ], dim=1).reshape((-1, config.hidden_size))
                        print(torch.equal(mg_model_state[key],attn_weight  ))
                    if "linear_qkv.bias" in key:
                        attn_weight = torch.cat([
                                hf_attn.q_proj.bias.reshape((num_query_groups, num_querys_per_group * dim)),
                                hf_attn.k_proj.bias.reshape((num_query_groups, dim)),
                                hf_attn.v_proj.bias.reshape((num_query_groups, dim)),
                            ], dim=1).reshape(-1)
                        print(torch.equal(mg_model_state[key],attn_weight  ))
                        #TODO:
                    if "linear_qkv.layer_norm_weight" in key:
                        print(torch.equal(mg_model_state[key], hf_layer.input_layernorm.weight    ))
            if "mlp" in key:
                hf_mlp = hf_layer.mlp
                if "pre_mlp_layernorm" in key:
                    print(torch.equal(mg_model_state[key], hf_layer.post_attention_layernorm.weight    ))
                if "router" in  key:
                    print(torch.equal(mg_model_state[key], hf_mlp.gate.weight   ))
                if "mlp.experts.linear_fc" in  key:
                    hf_experts = hf_mlp.experts
                    match = re.search(r'weight(\d+)$', key)
                    if match:
                        expert_idx = int(match.group(1))
                        global_id = local_expert_indices [expert_idx]
                    else:
                        print(f"⚠️  Skip non-indexed expert key: {key}")
                    if "experts.linear_fc1" in  key:
                        mg_weight = mg_model_state[key]
                        hf_weight =  torch.cat([
                                hf_experts[global_id].gate_proj.weight,
                                hf_experts[global_id].up_proj.weight
                            ], dim=0)
                        if not torch.equal(mg_weight, hf_weight):
                            print(f"    ❌ Mismatch in {key}, {get_param_name(hf_model,  hf_experts[global_id].gate_proj.weight) }, {get_param_name(hf_model,  hf_experts[global_id].up_proj.weight) }")
                            print(f"    Megatron weight shape: {mg_weight.shape}")
                            print(f"    HF weight shape      : {hf_weight.shape}")
                        else:
                            print(f"    ✅ Match in {key} {get_param_name(hf_model,  hf_experts[global_id].gate_proj.weight) },       {get_param_name(hf_model,  hf_experts[global_id].up_proj.weight) }")
                            
                    if "experts.linear_fc2" in  key:
                        mg_weight = mg_model_state[key]
                        hf_weight =   hf_experts[global_id].down_proj.weight
                        if not torch.equal(mg_weight, hf_weight):
                            print(f"    ❌ Mismatch in {key},   {get_param_name(hf_model,  hf_experts[global_id].down_proj.weight) }")
                            print(f"    Megatron weight shape: {mg_weight.shape}")
                            print(f"    HF weight shape      : {hf_weight.shape}")
                        else:
                            print(f"    ✅ Match in {key}  {get_param_name(hf_model,  hf_experts[global_id].down_proj.weight) }")
                            
                if "shared_experts"  in  key:
                    if "shared_experts.gate_weight"  in  key:
                        mg_weight = mg_model_state[key]
                        hf_weight = hf_mlp.shared_expert_gate.weight
                        if not torch.equal(mg_weight, hf_weight):
                            print(f"    ❌ Mismatch in {key},   {get_param_name(hf_model,hf_weight)  }")
                        else:
                            print(f"    ✅ Match in {key}  {get_param_name(hf_model,hf_weight)}")
                    if  "shared_experts.linear_fc1" in key:
                        mg_weight = mg_model_state[key]
                        hf_weight =  torch.cat([
                                hf_mlp.shared_expert.gate_proj.weight,
                                hf_mlp.shared_expert.up_proj.weight,
                            ], dim=0)
                        if not torch.equal(mg_weight, hf_weight):
                            print(f"    ❌ Mismatch in {key}")
                        else:
                            print(f"    ✅ Match in {key}" )
                    if  "shared_experts.linear_fc2" in key:        
                        mg_weight = mg_model_state[key]
                        hf_weight =  hf_mlp.shared_expert.down_proj.weight
                        if not torch.equal(mg_weight, hf_weight):
                            print(f"    ❌ Mismatch in {key}")
                        else:
                            print(f"    ✅ Match in {key}" )
                        
parser = argparse.ArgumentParser(description='Modify model checkpoint file paths.')
parser.add_argument('--root_dir', type=str,  help='Root directory containing model files', default="/mnt/data/mcore-TP1PP1EP4")
parser.add_argument('--hf_path', type=str,  help='Root directory containing model files', default="/mnt/data/Qwen1.5-MoE-A2.7B-Chat")
args = parser.parse_args()
root_dir = args.root_dir
hf_path =  args.hf_path
config =  AutoConfig.from_pretrained(hf_path)
print(config.qkv_bias)
 
hf_model = AutoModelForCausalLM.from_pretrained(hf_path , device_map="cpu")

target_name = 'model_optim_rng.pt'
if not os.path.exists(root_dir):
    print(f"not exist: {root_dir}")
    exit(0)
all_paths = []
for dirpath, dirnames, filenames in os.walk(root_dir):
    if target_name in filenames:
        full_path = os.path.join(dirpath, target_name)
        all_paths.append(full_path)

for path in all_paths[:]:
    print(path)
for path in all_paths[:]:
    print("\nChecking file:", path)
    check_hf(path)