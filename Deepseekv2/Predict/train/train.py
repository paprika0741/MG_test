import re
import ast
import torch.nn.functional as F
import os
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt
from megatron.core.transformer.moe.moe_utils import topk_softmax_with_capacity
import numpy as np
import pandas as pd
import torch.nn as nn
import sys
sys.path.append("../../../")  # 把项目根目录加入搜索路径
from utils.trace import  *
from argparse import Namespace
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import glob
from collections import defaultdict
import torch
args = Namespace(
    moe_expert_capacity_factor =None,
    moe_pad_expert_input_to_capacity = False,
    moe_token_drop_policy = "probs",
    moe_router_pre_softmax = True,
    moe_router_num_groups = None,
    moe_router_group_topk = None,
    moe_router_topk_scaling_factor =  1.0,
    deterministic_mode = False,
    moe_router_score_function = "softmax",
    moe_router_enable_expert_bias = False,
    moe_router_dtype = "fp32"
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trace_dir = "/home/download/models/mg_core/DeepSeek-V2-Lite/200_1/"
path = "/home/download/models/mg_core/DeepSeek-V2-Lite/mcore-TP1PP1EP4/iter_0000001/mp_rank_00_000/model_optim_rng.pt"
save_dir = "trained_gate"

input_data, num_layers,num_requests =  read_trace(trace_dir, "*input**.pt")
logits_data  ,_ ,_ = read_trace(trace_dir, "*logits**.pt")
map_data ,_ ,_ = read_trace(trace_dir, "*routing_map**.pt")

all_gate = load_gate(path)
print(input_data.head())
 

topk_per_token  =  get_tensor( map_data, 2, 0,0).sum(dim=1)
if not torch.all(topk_per_token == topk_per_token[0]):
    raise ValueError(f"Inconsistent topk per token: {topk_per_token.tolist()}")
topk = topk_per_token[0].item()
print("topk",topk)


request_set = set(input_data["request"].unique())
request_list = sorted(list(input_data["request"].unique()))
random.seed(42)  # 保证可复现
random.shuffle(request_list)
split_idx = int(len(request_list) * 0.9)
train_requests = set(request_list[:split_idx])
test_requests = set(request_list[split_idx:])
print(f"Total: {len(request_list)}, Train: {len(train_requests)}, Test: {len(test_requests)}")


os.makedirs(save_dir, exist_ok=True)  # 如果目录不存在则创建

record = [ ]
for step in range(1,5):
    for layer in range(1,num_layers):
        next_layer = layer + step
        if next_layer > num_layers:
            break
        gate_weight = all_gate[next_layer]
        model = TrainableRouter(gate_weight,args).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        train_losses = []
        eval_scores = []
        acc = eval(test_requests, model, topk,layer, next_layer,args,input_data,map_data)
        eval_scores.append(acc)
        for epoch in range(0,10):
            model.train()
            for request in train_requests:
                
                
                # === predict next layer output ===
                pre_layer_input = get_tensor(input_data, layer, request, 0 ).to(device).float()
                target_output = get_tensor(logits_data, next_layer, request, 0).to(device).float()
                pred = model(pre_layer_input)
                # logits =  pred.view(-1,  pred.shape[-1])
                # prob = torch.softmax(logits, dim=-1) 
                # real_routing_map = get_tensor(map_data,next_layer,request,0).to(device).float()
                # loss = F.binary_cross_entropy(prob, real_routing_map)
                # optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()
                # train_losses.append(loss.item())

                
                
                loss = loss_fn(pred, target_output)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
                print(f"[Request {request}] Loss: {loss.item():.6f}")
            
            # After training epoch
            acc = eval(test_requests, model,topk, layer, next_layer,args,input_data,map_data)
            eval_scores.append(acc)
        record.append({
        "layer": layer,
        "step": step,
        "acc": eval_scores,
        "loss":train_losses,
    
        })
        df = pd.DataFrame(record)
        csv_path = "result.csv"
        df.to_csv(csv_path, index=False)
        print(f"[Saved] All layer results saved to {csv_path}")
        print(f"Layer {layer} --> Layer {next_layer}",eval_scores )
    # save gate
        path = os.path.join(save_dir, f"layer_{layer}_next_layer{next_layer}_step{step}.pt")
        torch.save(model.state_dict(), path)
        print(f"[INFO] Saved trained gate weights to {path}")



        
    df = pd.DataFrame(record)
    csv_path = "result_loss_output.csv"
    df.to_csv(csv_path, index=False)
    print(f"[Saved] All layer results saved to {csv_path}")