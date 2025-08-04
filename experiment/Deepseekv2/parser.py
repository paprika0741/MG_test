import sys
sys.path.append("../../")  # 把项目根目录加入搜索路径
from utils.trace import  *
from argparse import Namespace
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_dir ="./bs8_in32_out16_num64"
input_data, num_layers,num_requests =  read_trace(save_dir, "*input**.pt")
logits_data  ,_ ,_ = read_trace(save_dir, "*logits**.pt")
map_data ,_ ,_ = read_trace(save_dir, "*routing_map**.pt")
path = "/home/download/models/mg_core/DeepSeek-V2-Lite/mcore-TP1PP1EP4/iter_0000001/mp_rank_00_000/model_optim_rng.pt"
all_gate = load_gate(path)
print(input_data.head())

def analyze_activate_expert_rate(map_data):
    # 统计 activate_expert_rate
    # sum across layer
    print(map_data.head())
    request_set = sorted(list(map_data["request"].unique()))
    print("Request IDs:", request_set)
    num_layer_set = sorted(list(map_data["layer"].unique()))
    print("Layer IDs:", num_layer_set)
    Activate_Rate_list = []
    for request in request_set:
 
        iters = map_data[
                (map_data["request"] == request) 
            ]["iter"].unique()
        for iter_id in iters:
            activated_experts = 0
            total_experts = 0
            for layer in num_layer_set:
                map_tensor = get_tensor(map_data, layer, request, iter_id)
                nonzero_count = torch.count_nonzero(map_tensor.sum(dim=0)).item()
                activated_experts += nonzero_count
                total_experts += map_tensor.shape[1]
                # print(nonzero_count,  map_tensor.shape[1])
            # print(activated_experts/total_experts)
            Activate_Rate_list.append(activated_experts/total_experts)
    # 
    print("Activate")
    print(len(Activate_Rate_list))
    # print(Activate_Rate_list)
    print(sum(Activate_Rate_list) / len(Activate_Rate_list))

def evaluate_routing_prediction_accuracy(args, input_data, map_data, all_gate, topk, step):
    print(map_data.head())
    request_set = sorted(list(map_data["request"].unique()))
    print("Request IDs:", request_set)
    num_layer_set = sorted(list(map_data["layer"].unique()))
    print("Layer IDs:", num_layer_set)
    # 存储所有 request-iter 对应的预测匹配率（accuracy）
    match_rate_list = []
    for request in request_set:
 
        iters = map_data[map_data["request"] == request]["iter"].unique()
        for iter_id in iters:
            matched_expert_count = 0 # 预测正确的专家数
            real_expert_count = 0 # 实际应激活的专家数
            for layer in num_layer_set:
                next_layer = layer + step
                if next_layer not in num_layer_set:
                    break
                # 获取当前 layer 的输入张量（用于模拟 gating 输出）
                pre_layer_input = get_tensor(input_data, layer, request, iter_id)
                 # 使用 gating 网络预测下一层的 gate 输出
                pred_next_layer_output = gating(pre_layer_input, all_gate[next_layer], args)
                pred_flat = pred_next_layer_output.view(-1, pred_next_layer_output.shape[-1])
                # 生成 top-k 路由结果（布尔 mask）以及其他输出
                _, pred_routing_map, _ = topk_softmax_with_capacity(
                    pred_flat,
                    topk=topk,
                    capacity_factor=args.moe_expert_capacity_factor,
                    pad_to_capacity=args.moe_pad_expert_input_to_capacity,
                    drop_policy=args.moe_token_drop_policy,
                    use_pre_softmax=args.moe_router_pre_softmax,
                    num_groups=args.moe_router_num_groups,
                    group_topk=args.moe_router_group_topk,
                    scaling_factor=args.moe_router_topk_scaling_factor,
                    deterministic_mode=args.deterministic_mode,
                    score_function=args.moe_router_score_function,
                    expert_bias=args.moe_router_enable_expert_bias,
                )
                # 获取真实的路由布尔 mask（ground-truth expert routing map）
                real_routing_map = get_tensor(map_data, next_layer, request, iter_id)
                # 被预测选择的 expert ID
                predicted_expert = (pred_routing_map.sum(0) != 0).nonzero(as_tuple=True)[0]
                # 被实际选择的 expert ID（非零列）
                real_expert = (real_routing_map.sum(0) != 0).nonzero(as_tuple=True)[0]
                matched = torch.isin(predicted_expert, real_expert)
                num_matched = matched.sum().item()
                matched_expert_count += num_matched
                real_expert_count += len(real_expert)
                # print(matched_expert_count, real_expert_count)
            match_rate = (
                        matched_expert_count / real_expert_count
                        if real_expert_count > 0
                        else 0.0  # 用 float 类型防止张量混入
                    )
            match_rate_list.append(match_rate )
    # 
    print("Predictor")
    print(len(match_rate_list))
    # print(match_rate_list)
    print(sum(match_rate_list) / len(match_rate_list))
    return match_rate_list

topk_per_token  =  get_tensor( map_data, 2, 0,0).sum(dim=1)
if not torch.all(topk_per_token == topk_per_token[0]):
    raise ValueError(f"Inconsistent topk per token: {topk_per_token.tolist()}")
topk = topk_per_token[0].item()
print("topk",topk)

analyze_activate_expert_rate(map_data)
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
evaluate_routing_prediction_accuracy(
    args, input_data, map_data,all_gate, topk, 1
)
print(save_dir)