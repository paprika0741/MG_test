import torch
import os
import argparse
import re
import json
path = "/home/download/models/mg_core/DeepSeek-V2-Lite/mcore-TP1PP1EP4/iter_0000001/mp_rank_00_000/model_optim_rng.pt"
args = torch.load(path, map_location="cpu", weights_only=False)["args"]
args_dict = vars(args)
keys = [
    "moe_expert_capacity_factor",
    "moe_pad_expert_input_to_capacity",
    "moe_token_drop_policy",
    "moe_router_pre_softmax",
    "moe_router_num_groups",
    "moe_router_group_topk",
    "moe_router_topk_scaling_factor",
    "deterministic_mode",
    "moe_router_score_function",
    "moe_router_enable_expert_bias"
]

for k in keys:
    print(f"{k}: {args_dict.get(k)}")
# 构造只保留能被序列化的字段
serializable_args = {}
for k, v in args_dict.items():
    try:
        json.dumps(v)
        serializable_args[k] = v
    except TypeError:
        print(f"[SKIP] Field '{k}' is not JSON serializable, skipped.")

# 保存为 JSON 文件
with open("args.json", "w") as f:
    json.dump(serializable_args, f, indent=4)