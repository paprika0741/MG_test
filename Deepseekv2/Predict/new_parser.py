import sys
from argparse import Namespace

sys.path.append("../../")  # 把项目根目录加入搜索路径
from utils.trace import  *
save_dir = "/home/download/models/mg_core/DeepSeek-V2-Lite/100_40/"
input_data, num_layers,num_requests =  read_trace(save_dir, "*input**.pt")
logits_data  ,_ ,_ = read_trace(save_dir, "*logits**.pt")
map_data ,_ ,_ = read_trace(save_dir, "*routing_map**.pt")

path = "/home/download/models/mg_core/DeepSeek-V2-Lite/mcore-TP1PP1EP4/iter_0000001/mp_rank_00_000/model_optim_rng.pt"
all_gate = load_gate(path)
print(input_data.head())

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
topk_per_token  =  get_tensor( map_data, 2, 0,0).sum(dim=1)
if not torch.all(topk_per_token == topk_per_token[0]):
    raise ValueError(f"Inconsistent topk per token: {topk_per_token.tolist()}")
topk = topk_per_token[0].item()
print("topk",topk)
 
result =  dict()
for step in range(1,5):
    data= get_prefilling_layer_predict(args,input_data, logits_data, map_data,all_gate,num_layers,topk=topk,step=step)
    result [step] = data
plot(result, "prefilling.pdf","Prefilling")
save_similarity(result,"prefilling.csv", "prefilling")

# result =  dict()
# for step in range(1,5):
#     data= get_decoding_layer_predict(args,input_data, logits_data, map_data,all_gate,num_layers,topk=topk,step=step)
#     result [step] = data
# plot(result, "decoding.pdf", "Decoding")
# save_similarity(result,"decoding.csv", "decoding")



# result =  dict()
# for step in range(1,5):
#     data= get_all_layer_predict(args,input_data, logits_data, map_data,all_gate,num_layers,topk=topk,step=step)
#     result [step] = data
# plot(result, "all.pdf","Prefilling+Decoding")
# save_similarity(result,"all.csv", "all")

