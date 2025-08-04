import sys
from argparse import Namespace
import matplotlib.pyplot as plt
import pickle

sys.path.append("../../../")  # 把项目根目录加入搜索路径
from utils.trace import  *
save_dir ="/home/download/models/mg_core/Qwen1.5-MoE-A2.7B-Chat/200_1/"
input_data, num_layers,num_requests =  read_trace(save_dir, "*input**.pt")
logits_data  ,_ ,_ = read_trace(save_dir, "*logits**.pt")
map_data ,_ ,_ = read_trace(save_dir, "*routing_map**.pt")

path = "/home/download/models/mg_core/Qwen1.5-MoE-A2.7B-Chat/mcore-TP1PP1EP4/iter_0000001/mp_rank_00_000/model_optim_rng.pt"
all_gate = load_gate(path)
print(input_data.head())

args = Namespace(
    moe_expert_capacity_factor =None,
    moe_pad_expert_input_to_capacity = False,
    moe_token_drop_policy = "probs",
    moe_router_pre_softmax = True,
    moe_router_num_groups = None,
    moe_router_group_topk = None,
    moe_router_topk_scaling_factor = None,
    deterministic_mode = False,
    moe_router_score_function = "softmax",
    moe_router_enable_expert_bias = False,
     moe_router_dtype = "fp32"
)
topk_per_token  =  get_tensor( map_data, 2, 0,0).sum(dim=1)
if not torch.all(topk_per_token == topk_per_token[0]):
    raise ValueError(f"Inconsistent topk per token: {topk_per_token.tolist()}")
topk = topk_per_token[0].item()
print("topk",topk)
def get_all_coe(args, input_data, logits_data, map_data, all_gate, num_layer, topk, step ):
    request_set = set(input_data["request"].unique())
    res = dict()
    for layer in range(1, num_layer):
        next_layer = layer + step
        if next_layer > num_layer:
            break
        pearson_corr_list = []
        for request in request_set:
            iters = input_data[
                (input_data["layer"] == layer) &
                (input_data["request"] == request) &
                (input_data["iter"]  == 0 )
            ]["iter"].unique()
            for iter_id in iters:
            # === predict next layer output ===
                pre_layer_input = get_tensor(input_data, layer, request, iter_id)
                pred_nex_layer_output = gating(pre_layer_input, all_gate[next_layer],args)
                pred_nex_layer_output =  pred_nex_layer_output.view(-1,  pred_nex_layer_output.shape[-1])
                _, pred_routing_map, _ = topk_softmax_with_capacity(
                    pred_nex_layer_output,
                    topk = topk ,
                    capacity_factor= args.moe_expert_capacity_factor  ,
                    pad_to_capacity= args.moe_pad_expert_input_to_capacity,
                    drop_policy=args.moe_token_drop_policy,
                    use_pre_softmax=args.moe_router_pre_softmax,
                    num_groups=args.moe_router_num_groups,
                    group_topk=args.moe_router_group_topk,
                    scaling_factor=args.moe_router_topk_scaling_factor,
                    deterministic_mode=args.deterministic_mode,
                    score_function=args.moe_router_score_function,
                    expert_bias= args.moe_router_enable_expert_bias,
                )
                real_routing_map = get_tensor(map_data,next_layer,request,iter_id)
                pred_distribution = pred_routing_map.sum(axis=0)
                real_distribution = real_routing_map.sum(axis=0)
                pred_np = pred_distribution.detach().cpu().numpy()
                real_np = real_distribution.detach().cpu().numpy()
                corr_matrix = np.corrcoef(pred_np, real_np)
                pearson_corr = corr_matrix[0, 1]
                print("Pearson correlation coefficient:", pearson_corr)
                pearson_corr_list.append(pearson_corr)
        res[layer] = pearson_corr_list
    return  res
step = 1
res = get_all_coe(args, input_data, logits_data, map_data, all_gate,  num_layers, topk, step, )
print(len(res))
print(len(res[1]))
fig, axes = plt.subplots(6, 5, figsize=(20, 18))  # 6 行 5 列，共 30 个子图
axes = axes.flatten()  # 转成 1D 数组，方便访问

for i in range(23):  # 遍历索引 0 到 25
    layer = i + 1
    axes[i].hist(res[layer], bins=50, edgecolor='black')
    axes[i].set_title(f'$L_{{{layer}}} \\rightarrow L_{{{layer+step}}}$')
    # axes[i].set_xlabel('Value')
    axes[i].set_xlabel("Pearson correlation coefficient")
    axes[i].set_ylabel('Freq')
    axes[i].grid(True, linestyle='--', alpha=0.5)

with open('iteration_prefilling_200_1.pkl', 'wb') as f:
    pickle.dump(res, f)
for j in range(23, len(axes)):
    axes[j].axis('off')
plt.tight_layout()
plt.savefig("iteration_prefilling_200_1.pdf")