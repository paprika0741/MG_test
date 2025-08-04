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
 

step =  0
for req in range(1):
    for layer in range(2,5):
        next_layer = layer + step
        if next_layer > num_layers:
            break
        input_tensor = get_tensor(input_data, layer, req, 0 )
        output = gating( input_tensor, all_gate[next_layer],args )
        output_tensor = get_tensor(logits_data, next_layer, req, 0 )
        diff = torch.abs(output - output_tensor)
        print("最大差异:", diff.max())
        sim = get_similarity(output,output_tensor)
        print(sim.mean().item())
        
        _, re_routing_map, _ = topk_softmax_with_capacity(
            output_tensor.view(-1,  output_tensor.shape[-1]) ,
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
        real_routing_map = get_tensor(map_data,next_layer,req,0)
        print(torch.equal(real_routing_map,re_routing_map))
        break
print("=============")
layer =6
next_layer = 7
request_id = 1
print(f"[Layer {layer} → {next_layer}] Request ID: {request_id}")

# 获取 input_tensor 并打印形状
input_tensor = get_tensor(input_data, layer, request_id, 0)
print("Input tensor shape:", input_tensor.shape,input_tensor.dtype )

# gating 预测
pred_output = gating(input_tensor, all_gate[next_layer], args)
print("Predicted output shape:", pred_output.shape, pred_output.dtype)

# 获取 ground truth output tensor
output_tensor = get_tensor(logits_data, next_layer, request_id, 0)
print("Ground truth output shape:", output_tensor.shape,output_tensor.dtype )

# 计算差异和相似度
diff = torch.abs(pred_output - output_tensor)
print("最大差异:", diff.max().item())

sim = get_similarity(pred_output, output_tensor)
print("Average similarity score between pred_output and output_tensor:", sim.mean().item())

# 基于 pred_output 生成预测路由图
print("\nRouting with pred_output...")
_, pred_routing_map, _ = topk_softmax_with_capacity(
    pred_output.view(-1, pred_output.shape[-1]),
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

# 与真实 map 对比
real_routing_map = get_tensor(map_data, next_layer, request_id, 0)
correct_map = pred_routing_map & real_routing_map
total_correct = correct_map.sum().item()
total_expert = real_routing_map.sum().item()
avg_acc = total_correct / total_expert
print(f"Pred Output Routing Accuracy: {total_correct} / {total_expert} = {avg_acc:.4f}")

# 再用真实 output tensor 测一次 routing
print("\nRouting with real output_tensor...")
_, re_routing_map, _ = topk_softmax_with_capacity(
    output_tensor.view(-1, output_tensor.shape[-1]),
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

correct_map = re_routing_map & real_routing_map
total_correct = correct_map.sum().item()
total_expert = real_routing_map.sum().item()
avg_acc = total_correct / total_expert
print(f"Real Output Routing Accuracy: {total_correct} / {total_expert} = {avg_acc:.4f}")

######################333
print(pred_routing_map.shape)
print(real_routing_map.shape)