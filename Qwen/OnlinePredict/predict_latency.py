from megatron.core.transformer.moe.moe_utils import topk_softmax_with_capacity
import torch
from argparse import Namespace
import time
import matplotlib.pyplot as plt
import numpy as np
args = Namespace(
    topk = 4,
    num_moe_experts = 60,
    hidden_size = 2048,
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

predictor = torch.nn.Parameter(
    torch.empty(
        (args.num_moe_experts, args.hidden_size),
        device=torch.cuda.current_device(),
        dtype=torch.bfloat16
    )
)
print(predictor.device)
memory_bytes = predictor.element_size() * predictor.numel()
memory_MB = memory_bytes / (1024 ** 2)

print(f"Predictor memory usage: {memory_MB:.2f} MB")
 
@torch.no_grad()
def predict( input: torch.Tensor):
    if predictor.device.type == 'cpu':
        predictor.data =  predictor.data.to(device=torch.cuda.current_device())
    router_dtype = torch.float32
    logits = torch.nn.functional.linear(input.to(router_dtype), predictor.to(router_dtype))
    pred_nex_layer_output = logits.view(-1,logits.shape[-1])
    _, pred_routing_map, _ = topk_softmax_with_capacity(
                pred_nex_layer_output,
                topk = args.topk ,
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
res = []

for token in range(1,4097):
    print(token)
    input_data = torch.rand(token, 1, args.hidden_size, dtype=torch.bfloat16, device='cuda')
    # Warm-up
    print("Warm up")
    for _ in range(20):
        _ = predict(input_data)
    # 正式计时
    N = 20
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(N):
        _ = predict(input_data)
    torch.cuda.synchronize()
    end = time.time()
    avg_time_ms = (end - start) * 1000 / N
    res.append(avg_time_ms)
    print(f"Average Predict time: {avg_time_ms:.3f} ms (over {N} runs)")
print(res)
plt.figure(figsize=(10, 6))
token_lengths = list(range(1, 4097))

plt.plot(token_lengths, res, label='Average Predict Time', linewidth=2)
plt.xlabel('Input Token Length', fontsize=12)
plt.ylabel('Average Predict Time (ms)', fontsize=12)
# plt.title('Prefill Latency vs. Input Token Length', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("prefill_latency_vs_token_length.pdf")
plt.close()
