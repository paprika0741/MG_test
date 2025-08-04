import torch
import os
import torch.serialization
import argparse
torch.serialization.add_safe_globals([argparse.Namespace])

# 遍历 rank
for rank in [0, 1, 2, 3]:
    path_a = f"/home/download/models/mg_core/Qwen1.5-MoE-A2.7B-Chat_test/mcore-TP1PP1EP4/iter_0000001/mp_rank_00_00{rank}/model_optim_rng.pt"
    path_b = f"/home/download/models/mg_core/Qwen1.5-MoE-A2.7B-Chat/mcore-TP1PP1EP4/iter_0000001/mp_rank_00_00{rank}/model_optim_rng.pt"

    # 加载模型 state_dict
    state_a = torch.load(path_a, map_location="cpu", weights_only=False)["model"]
    state_b = torch.load(path_b, map_location="cpu", weights_only=False)["model"]

    print(f"Comparing rank {rank}...")

    for key in state_a:
        val_a = state_a[key]
        val_b = state_b.get(key, None)

        if val_a is None:
            print(f"  {key} is None in model A")
        elif val_b is None:
            print(f"  {key} missing in model B")
        elif not torch.equal(val_a, val_b):
            print(f"  🔁 Mismatch in key: {key}")
