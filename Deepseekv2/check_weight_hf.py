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
    expert_indices = list(range(config.n_routed_experts))  # 所有 expert 的 index
    # 每个 rank 应该负责的 expert 数量
    num_local_experts = int(config.n_routed_experts // world_size)
    # 当前 rank 负责的 expert index 范围
    start_idx = ep_rank * num_local_experts
    end_idx = start_idx + num_local_experts
    # 当前 rank 实际负责的 expert indices
    local_expert_indices = expert_indices[start_idx:end_idx]
    # 打印调试信息
    print(f"Total experts: {config.n_routed_experts}")
    print(f"World size (EP): {world_size}")
    print(f"EP rank: {ep_rank}")
    print(f"Experts per rank: {num_local_experts}")
    print(f"Local expert indices: {local_expert_indices}")
    print("====================HF weight=====================")
    for name, param in hf_model.named_parameters():
        print(f"{name}: {param.shape}")
    print("====================MG weight=====================")
    for key, value in mg_model_state.items():
        if value is None:
            print(key, "is none")
        else:
            print(key, value.shape)
        
        if "_extra_state" in key:
            print("pass")
            continue
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
                if "linear_q_proj" in key:
                    print(torch.equal(mg_model_state[key], hf_attn.q_proj.weight    ))
                if "linear_kv_down_proj" in key:
                    print(torch.equal(mg_model_state[key], hf_attn.kv_a_proj_with_mqa.weight    ))
                if "linear_kv_up_proj" in key:
                    if "linear_kv_up_proj.weight" in key:
                        print(torch.equal(mg_model_state[key], hf_attn.kv_b_proj.weight    ))
                    elif "linear_kv_up_proj.layer_norm_weight" in key:
                        print(torch.equal(mg_model_state[key], hf_attn.kv_a_layernorm.weight    ))
                    else:
                        raise ValueError(f"Unrecognized key : {key}") 
            elif  "input_layernorm" in key:
                print(torch.equal(mg_model_state[key], hf_layer.input_layernorm.weight))
                
            
            elif layer_idx == 0 and "mlp" in key:
                print("Dense layer")
                hf_mlp = hf_layer.mlp
                if "mlp.linear_fc1." in key:
                    if "mlp.linear_fc1.layer_norm_weight" in key:
                        print(torch.equal(mg_model_state[key], hf_layer.post_attention_layernorm.weight))
                    elif "mlp.linear_fc1.weight" in key:
                        hf_weight =  torch.cat([
                            hf_mlp.gate_proj.weight,
                            hf_mlp.up_proj.weight,
                        ], dim=0)
                        print(torch.equal(mg_model_state[key], hf_weight ))
                    else:
                        raise ValueError(f"Unrecognized key : {key}") 
                elif "mlp.linear_fc2.weight" in key:
                    hf_weight =   hf_mlp.down_proj.weight
                    print(torch.equal(mg_model_state[key], hf_weight ))
                else:
                    raise ValueError(f"Unrecognized key : {key}") 

            elif layer_idx != 0 and "mlp" in key:
                hf_mlp = hf_layer.mlp
                print("Sparse layer")
                assert layer_idx != 0 
                if "pre_mlp_layernorm" in key:
                    print(torch.equal(mg_model_state[key], hf_layer.post_attention_layernorm.weight    ))
                elif "router" in  key:
                    print(torch.equal(mg_model_state[key], hf_mlp.gate.weight   ))
                elif "mlp.experts.linear_fc" in key:
                    # experts
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
                            
                    elif "experts.linear_fc2" in  key:
                        mg_weight = mg_model_state[key]
                        hf_weight =   hf_experts[global_id].down_proj.weight
                        if not torch.equal(mg_weight, hf_weight):
                            print(f"    ❌ Mismatch in {key},   {get_param_name(hf_model,  hf_experts[global_id].down_proj.weight) }")
                            print(f"    Megatron weight shape: {mg_weight.shape}")
                            print(f"    HF weight shape      : {hf_weight.shape}")
                        else:
                            print(f"    ✅ Match in {key}  {get_param_name(hf_model,  hf_experts[global_id].down_proj.weight) }")
                    else:
                        raise ValueError(f"Unrecognized key : {key}") 
                elif "shared_experts" in  key:
                    if  "shared_experts.linear_fc1" in key:
                        mg_weight = mg_model_state[key]
                        hf_weight =  torch.cat([
                                hf_mlp.shared_experts.gate_proj.weight,
                                hf_mlp.shared_experts.up_proj.weight,
                            ], dim=0)
                        if not torch.equal(mg_weight, hf_weight):
                            print(f"    ❌ Mismatch in {key}")
                        else:
                            print(f"    ✅ Match in {key}" )
                    elif  "shared_experts.linear_fc2" in key:        
                        mg_weight = mg_model_state[key]
                        hf_weight =  hf_mlp.shared_experts.down_proj.weight
                        if not torch.equal(mg_weight, hf_weight):
                            print(f"    ❌ Mismatch in {key}")
                        else:
                            print(f"    ✅ Match in {key}" )
                        
parser = argparse.ArgumentParser(description='Modify model checkpoint file paths.')
parser.add_argument('--root_dir', type=str,  help='Root directory containing model files', default="/home/ec2-user/CodeSpace/NEW_Megatron/Megatron-LM-core_v0.12.0/Deepseekv2/mcore-TP1PP1EP4Layer2")
parser.add_argument('--hf_path', type=str,  help='Root directory containing model files', default="/mnt/data/DeepSeek-V2-Lite")
args = parser.parse_args()
root_dir = args.root_dir
hf_path =  args.hf_path
config =  AutoConfig.from_pretrained(hf_path, trust_remote_code=True )
 
 
hf_model = AutoModelForCausalLM.from_pretrained(hf_path , device_map="cpu",  trust_remote_code=True  )

target_name = 'model_optim_rng.pt'
if not os.path.exists(root_dir):
    print(f"not exist: {root_dir}")
    exit(0)
all_paths = []
for dirpath, dirnames, filenames in os.walk(root_dir):
    if target_name in filenames:
        full_path = os.path.join(dirpath, target_name)
        all_paths.append(full_path)
for path in all_paths[:1]:
    print("\nChecking file:", path)
    check_hf(path)
