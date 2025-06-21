import torch
import os
import re
import argparse
from transformers import AutoModelForCausalLM
from transformers  import AutoConfig
from collections import OrderedDict
def get_param_name(model, target_param):
    for name, param in model.named_parameters():
        if param is target_param:
            return name
    return None  # 如果找不到，返回 None
parser = argparse.ArgumentParser(description='Modify model checkpoint file paths.')
parser.add_argument('--root_dir', type=str,  help='Root directory containing model files', default="/home/download/models/mg_core/Qwen1.5-MoE-A2.7B-Chat/mcore-TP1PP1EP4")
parser.add_argument('--hf_path', type=str,  help='Root directory containing model files', default="/home/download/models/Qwen1.5-MoE-A2.7B-Chat")
args = parser.parse_args()
root_dir = args.root_dir
hf_path =  args.hf_path
config =  AutoConfig.from_pretrained(hf_path)
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
for path in all_paths:
    print(path)
def modify_keys(path):
    print("=" * 80)
    print(f"[Start] Processing: {path}")
    state  = torch.load(path,map_location="cpu", weights_only=False)
    print("[Info] Keys in checkpoint:")
    print("  ▸", list(state.keys()))
    model_state = state["model"]
    print(f"[Info] Total original model keys: {len(model_state)}")
    
    new_model_state = OrderedDict()
    
    for key, value in  model_state.items():
        if value is None:
            print(f"[Skip] {key} is None")
            continue
        print(f"[Process] {key}: shape = {value.shape}")
        if "experts.local_experts." in key  :
            parts = key.split(".")
            layer_idx = parts[2]
            expert_idx = parts[6]
            fc_name = parts[7]  # linear_fc1 or linear_fc2
            param_type = parts[-1]  # weight or _extra_state     
            if  "_extra_state" not in key:
            # "decoder.layers.0.mlp.experts.local_experts.0.linear_fc1.weight", --> "decoder.layers.0.mlp.experts.linear_fc1.weight0"
            # 构造新 key
                new_key = f"decoder.layers.{layer_idx}.mlp.experts.{fc_name}.weight{expert_idx}"
            else: 
                new_key = f"decoder.layers.{layer_idx}.mlp.experts.{fc_name}._extra_state"
            new_model_state[new_key] = value  
            print(f"  ↳ [Rename] {key} → {new_key}")
        
        elif "q_layernorm" in key and  "_extra_state" not in key :
        # decoder.layers.0.self_attention.q_layernorm.weight: torch.Size([2048])
        # decoder.layers.0.self_attention.q_layernorm._extra_state: torch.Size([5])
        # copy weight from HF model 
            layer_idx = int(key.split(".")[2])
            hf_weight = hf_model.model.layers[layer_idx].self_attn.q_norm.weight
            print(f"  ↳ [Replace from HF] q_layernorm in layer {layer_idx}, shape = {hf_weight.shape}")
            new_model_state[key] = hf_weight
        elif "k_layernorm" in key  and  "_extra_state" not in key :
        # decoder.layers.0.self_attention.k_layernorm.weight: torch.Size([2048])
        # decoder.layers.0.self_attention.k_layernorm._extra_state: torch.Size([5])
        # copy weight from HF model
            layer_idx = int(key.split(".")[2])
            hf_weight = hf_model.model.layers[layer_idx].self_attn.k_norm.weight
            print(f"  ↳ [Replace from HF] k_layernorm in layer {layer_idx}, shape = {hf_weight.shape}")
            new_model_state[key] = hf_weight
        else:
            # Direct copy
            print(f"  ↳ [Direct copy] {key}: shape = {value.shape}")
            new_model_state[key] = value  
    print(f"[Done] Writing modified state back to: {path}")
    state["model"] = new_model_state
    torch.save(state,path)
    print("=" * 80 + "\n")
    return new_model_state

 
def check_hf_weight(path):
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
    state =  torch.load(path,map_location="cpu", weights_only=False)
    mg_model_state = state["model"]
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
                        print("     Update...")
                        mg_model_state[key] = attn_weight
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
                            print("     Update...")
                            mg_model_state[key] = hf_weight
                        else:
                            print(f"    ✅ Match in {key} {get_param_name(hf_model,  hf_experts[global_id].gate_proj.weight) },       {get_param_name(hf_model,  hf_experts[global_id].up_proj.weight) }")
                            
                    if "experts.linear_fc2" in  key:
                        mg_weight = mg_model_state[key]
                        hf_weight =   hf_experts[global_id].down_proj.weight
                        if not torch.equal(mg_weight, hf_weight):
                            print(f"    ❌ Mismatch in {key},   {get_param_name(hf_model,  hf_experts[global_id].down_proj.weight) }")
                            print(f"    Megatron weight shape: {mg_weight.shape}")
                            print(f"    HF weight shape      : {hf_weight.shape}")
                            print("     Update...")
                            mg_model_state[key] = hf_weight
                        else:
                            print(f"    ✅ Match in {key}  {get_param_name(hf_model,  hf_experts[global_id].down_proj.weight) }")
                            
                if "mlp.shared_experts"  in  key:
                    if "shared_experts.gate_weight"  in  key:
                        mg_weight = mg_model_state[key]
                        hf_weight = hf_mlp.shared_expert_gate.weight
                        if not torch.equal(mg_weight, hf_weight):
                            print(f"    ❌ Mismatch in {key},   {get_param_name(hf_model,hf_weight)  }")
                            print("     Update...")
                            mg_model_state[key] = hf_weight 
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
                            print("     Update...")
                            mg_model_state[key] = hf_weight 
                        else:
                            print(f"    ✅ Match in {key}" )
                    if  "shared_experts.linear_fc2" in key:        
                        mg_weight = mg_model_state[key]
                        hf_weight =  hf_mlp.shared_expert.down_proj.weight
                        if not torch.equal(mg_weight, hf_weight):
                            print(f"    ❌ Mismatch in {key}")
                            print("     Update...")
                            mg_model_state[key] = hf_weight 
                        else:
                            print(f"    ✅ Match in {key}" )
    print("save to...", path)
    torch.save(state,path)                     

for i in all_paths:
    modify_keys(i)
    check_hf_weight(i)
 