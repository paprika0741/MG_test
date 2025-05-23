import os
import torch
import random
from collections import Counter
import eplb
import numpy as np
TOKEN_NUM = 1000  # 例如设为1000
NUM_EXPERTS = 8
TOPK = 2  # 每行有两个True
# 初始化为全False的布尔张量
routing_map = torch.zeros((TOKEN_NUM, NUM_EXPERTS), dtype=torch.bool)
# 为每一行随机选择TOPK个位置置True
for i in range(TOKEN_NUM):
    indices = torch.randperm(NUM_EXPERTS)[:TOPK]
    routing_map[i, indices] = True
# print(routing_map)
print(routing_map.sum(dim=0))
def get_imbalanced_routing_map(routing_map: torch.Tensor, expert_id: int, enforce_row_count: int):
    # modify routing_map in-place
    token_num, num_experts = routing_map.shape
    assert 0 <= expert_id < num_experts
    if enforce_row_count > token_num:
        enforce_row_count = token_num
    for i in range(enforce_row_count):
        row = routing_map[i]
        if not row[expert_id]:
            # 找出当前为 True 的 expert 中的一个，排除 expert_id（避免替换到它）
            true_indices = row.nonzero(as_tuple=True)[0]
            replace_idx = true_indices[torch.randint(len(true_indices), (1,)).item()]
            row[replace_idx] = False
            row[expert_id] = True
            routing_map[i] = row  # 写回
    return routing_map
get_imbalanced_routing_map(routing_map, 0, 501)
print(routing_map.sum(dim=0))

weight = routing_map.sum(dim=0).unsqueeze(0) 
print(weight.shape)
expert_num = weight.shape[1]
num_groups = expert_num     # 专家组数
num_nodes = 1      # 节点数
num_gpus = 4     # GPU总数
phy2log, log2phy, logcnt = eplb.rebalance_experts(weight, expert_num + num_gpus ,
                                                  num_groups, num_nodes, num_gpus)
print("phy2log",phy2log)
print("phy2log",phy2log.shape)  

from collections import defaultdict
def split_list_evenly(lst, C):
    N = len(lst)
    base = N // C
    remainder = N % C  # extra elements to distribute to the first `remainder` chunks

    result = []
    start = 0
    for i in range(C):
        end = start + base + (1 if i < remainder else 0)
        result.append(lst[start:end])
        start = end
    print("==============")
    print(len(lst), " --> ", end = " ")
    for i in result:
        print(len(i), end = " ")
    print(" ")
 
    return result

def get_new_data( original_indices, new_indices, map):
    TOKEN_NUM, NUM_EXPERTS = map.shape
    assert len(original_indices) == NUM_EXPERTS
    counts = Counter(new_indices)  # how many times each expert is reused
    usage_tracker = defaultdict(int)  # expert_id -> how many times already used
    
    print(len(new_indices))
    print(counts)
    results = []
    for i, global_expert_id in enumerate(new_indices):
        col = routing_map[:, global_expert_id]  # original expert col
        col_new = torch.zeros_like(col, dtype=torch.bool)
        if counts[global_expert_id] == 1:
            col_new = col.clone()
        else:
            # Split true indices evenly across repeated uses
            true_indices = torch.nonzero(col, as_tuple=True)[0].tolist()  # convert to list
            split_true_indices = split_list_evenly(true_indices,  counts[global_expert_id])
            chosen_indices  =  split_true_indices [usage_tracker[global_expert_id]]  
            col_new[chosen_indices] = True
            usage_tracker[global_expert_id] += 1 
        results.append(col_new.unsqueeze(1))
    final_result = torch.cat(results, dim=1)
    

    old_sum = routing_map.sum(dim=0).tolist()
    new_sum = final_result.sum(dim=0).tolist()
    # 打印原始专家的 token 分布和标准差
    print("Original token count per expert (before EPLB):")
    print(f"  Values : {old_sum}")
    print(f"  STD    : {np.std(old_sum):.2f}")

    # 打印 EPLB 映射后的 expert slot 分布和标准差
    print("Token count per output slot (after EPLB):")
    print(f"  Values : {new_sum}")
    print(f"  STD    : {np.std(new_sum):.2f}")

    new_sum_check = [0] * NUM_EXPERTS
    for i in range (0, len(new_indices)):
        id = new_indices[i]
        cnt = new_sum [i]
        new_sum_check[id] +=  cnt
    print(new_sum_check)
    row_sum = final_result.sum(dim=1)
    if not torch.all(row_sum == 2):
        wrong = torch.nonzero(row_sum != 2, as_tuple=True)[0]
        print(f"❌ Tokens not assigned to exactly 2 experts: {wrong.tolist()}")
    return old_sum,new_sum
original_indices = list(range(0,NUM_EXPERTS))
new_indices =  phy2log.flatten().tolist()
old_sum,new_sum = get_new_data(
    original_indices, new_indices,routing_map
)


 