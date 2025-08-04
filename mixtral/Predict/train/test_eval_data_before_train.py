import sys
import random
sys.path.append("../../../")  # 把项目根目录加入搜索路径
from utils.trace import  *
from argparse import Namespace
# save_dir = "./saved_iter_20"
save_dir = "/home/download/models/mg_core/Mixtral-8x7B-v0.1/200_1/"
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
# 假设 input_data、logits_data、map_data 都是 pandas.DataFrame
input_data = input_data[input_data["request"].isin(test_requests)]
logits_data = logits_data[logits_data["request"].isin(test_requests)]
map_data = map_data[map_data["request"].isin(test_requests)]
print(f"Filtered input_data: {len(input_data)} rows")
print(f"Filtered logits_data: {len(logits_data)} rows")
print(f"Filtered map_data: {len(map_data)} rows")
 
path = "/home/download/models/mg_core/Mixtral-8x7B-v0.1/mcore-TP1PP1EP4/iter_0000001/mp_rank_00_000/model_optim_rng.pt"
all_gate = load_gate(path)


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
 
expert_budget_res = dict()
for topk in [2,3,4,5]:
    result =  dict()
    print("budget ", topk)
    for step in range(1,5):

        data= get_prefilling_layer_predict(args,input_data, logits_data, map_data, all_gate, num_layers,topk=topk,step=step)
        result [step] = data
    plot(result, f"prefilling_expert_budget{topk}.pdf","prefilling")
    expert_budget_res[topk] =  result
plot_expert_budget(expert_budget_res,"expert_acc.pdf" )
plot_expert_budget_by_step(expert_budget_res,  "token_expert_accuracy")
