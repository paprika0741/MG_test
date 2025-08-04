import sys
import random
import pickle
sys.path.append("../../../")  # 把项目根目录加入搜索路径
from utils.trace import  *
from argparse import Namespace
def  get_prefilling_layer_predict_v2(args,input_data, logits_data,
                                 map_data,all_gate,num_layer,topk,step, ):
    request_set = set(input_data["request"].unique())
    avg_input_similarities = []
    avg_output_similarities = []
    token_expert_predict_accu = []
    for layer in range(1, num_layer):
        next_layer = layer + step
        if next_layer > num_layer:
            break
        out_similarity_list = []
        layer_predict_accu = []
        for request in request_set:
            # === predict next layer output ===
            pre_layer_input = get_tensor(input_data, layer, request, 0 )
            pred_nex_layer_output = gating( pre_layer_input, all_gate[next_layer],args )
            # token 
            # predicted next layer's expert selection
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
            real_routing_map = get_tensor(map_data,next_layer,request,0)
            correct_map = pred_routing_map & real_routing_map 
            total_correct  = correct_map.sum()
            total_expert = real_routing_map.sum()
            avg_acc = total_correct.float() / total_expert.float()
            layer_predict_accu.append(avg_acc.item())
        mean_acc = sum(layer_predict_accu) / len(layer_predict_accu) if layer_predict_accu else 0.0
        print(f"[ROUTING ACC] Layer {layer} → {next_layer}: {mean_acc:.4f}")
        token_expert_predict_accu.append(mean_acc)
        
    return {
        "input_similarities" : avg_input_similarities,
        "output_similarities": avg_output_similarities,
        "token_expert_predict_accuracy":token_expert_predict_accu
        } 
# save_dir = "./saved_iter_20"
save_dir = "/home/download/models/mg_core/DeepSeek-V2-Lite/200_1/"
input_data, num_layers,num_requests =  read_trace(save_dir, "*input**.pt")
logits_data  ,_ ,_ = read_trace(save_dir, "*logits**.pt")
map_data ,_ ,_ = read_trace(save_dir, "*routing_map**.pt")

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
 
print(f"Filtered input_data: {len(input_data)} rows")
print(f"Filtered logits_data: {len(logits_data)} rows")
print(f"Filtered map_data: {len(map_data)} rows")
path = "/home/download/models/mg_core/DeepSeek-V2-Lite/mcore-TP1PP1EP4/iter_0000001/mp_rank_00_000/model_optim_rng.pt"
all_gate = load_gate(path)
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

expert_budget_res = dict()
for topk in range(6,65):
    result =  dict()
    print("budget ", topk)
    for step in range(1,2):
        data= get_prefilling_layer_predict_v2(args,input_data, logits_data, map_data, all_gate, num_layers,topk=topk,step=step)
        result [step] = data
    # plot(result, f"prefilling_expert_budget{topk}.pdf","prefilling")
    expert_budget_res[topk] =  result


with open("expert_budget_res.pkl", "wb") as f:
    pickle.dump(expert_budget_res, f)

plot_expert_budget(expert_budget_res,"expert_acc.pdf" )
