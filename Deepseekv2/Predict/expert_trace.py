import sys
sys.path.append("../../")  # 把项目根目录加入搜索路径
from utils.trace import  *
from argparse import Namespace


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
save_dir = "/home/download/models/mg_core/DeepSeek-V2-Lite/200_1/"
input_data, num_layers,num_requests =  read_trace(save_dir, "*input**.pt")
logits_data  ,_ ,_ = read_trace(save_dir, "*logits**.pt")
map_data ,_ ,_ = read_trace(save_dir, "*routing_map**.pt")

path = "/home/download/models/mg_core/DeepSeek-V2-Lite/mcore-TP1PP1EP4/iter_0000001/mp_rank_00_000/model_optim_rng.pt"
all_gate = load_gate(path)
print(input_data.head())

topk_per_token  =  get_tensor( map_data, 2, 0,0).sum(dim=1)
if not torch.all(topk_per_token == topk_per_token[0]):
    raise ValueError(f"Inconsistent topk per token: {topk_per_token.tolist()}")
topk = topk_per_token[0].item()
print("topk",topk)
request_set = set(input_data["request"].unique())

layer = 2
next_layer = layer + 1
req = 0
iter = 0
layer_L_map = get_tensor(map_data,layer, req, iter)
layer_L1_map = get_tensor(map_data,next_layer, req, iter)
E = layer_L_map.shape[1]
# 枚举所有 token 的 (e, e') 对组合，统计共现频率
joint_counts = torch.zeros((E, E), dtype=torch.float32)
# 遍历所有 token，统计 (e, e') 共现
# 在 Layer L 中选中 expert e 的同时，Layer L+1 中选中 expert e' 的概率
for l_mask, l1_mask in zip(layer_L_map, layer_L1_map):
    experts_L = torch.nonzero(l_mask, as_tuple=False).squeeze(1)
    experts_L1 = torch.nonzero(l1_mask, as_tuple=False).squeeze(1)
    for e in experts_L:
        for e1 in experts_L1:
            joint_counts[e, e1] += 1

# 转换成联合概率
total_pairs = joint_counts.sum().item()
joint_probs = joint_counts / total_pairs

# 边缘分布 P(e) = sum over e1 of P(e, e1)
marginal_probs = joint_probs.sum(dim=1, keepdim=True)  # shape: [E, 1]

# 计算 H(E_L1 | E_L)
eps = 1e-12
conditional_entropy = - (joint_probs * (torch.log2(joint_probs + eps) - torch.log2(marginal_probs + eps))).sum()

print(f"Conditional Entropy H(E_L+1 | E_L): {conditional_entropy.item():.4f} bits")
########
# 每一层记录的是 expert 集合

req = 0
iter = 0
T,E =  get_tensor(map_data,2, req, iter).shape 
print(T,E)
trace = [[] for _ in range(T)]  # trace[token] = list of expert sets (per layer)
for layer in range(2,  27):
    layer_map = get_tensor(map_data, layer, req, iter)  # shape [T, E]
    for token in range(T):
        experts = torch.nonzero(layer_map[token], as_tuple=False).squeeze(1)
        trace[token].append(experts.tolist())
print(len(trace))
print(len(trace[0]))
print(len(trace[0][0]))

from collections import defaultdict, Counter

prefix_tree = defaultdict(Counter)
L = 3
for t in trace:
    if len(t) <= L:
        continue  # prefix 太短，跳过

    prefix = tuple(tuple(sorted(e)) for e in t[:L])     # 哈希化 prefix
    next_experts = t[L]  # 下一层的 expert 列表
    for e in next_experts:
        prefix_tree[prefix][e] += 1
query_prefix = trace[0][:L]  # 用第0个 token 的前L层作为测试
prefix_key = tuple(tuple(sorted(e)) for e in query_prefix)

if prefix_key in prefix_tree:
    counter = prefix_tree[prefix_key]
    total = sum(counter.values())
    prob = {e: v / total for e, v in counter.items()}
    print(f"P(E_{L+1} | prefix) = {prob}")
else:
    print("Prefix not seen in training data.")