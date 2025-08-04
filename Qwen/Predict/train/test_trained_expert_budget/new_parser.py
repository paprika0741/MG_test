import re
import ast
import torch.nn.functional as F
import os
from argparse import Namespace
import matplotlib.pyplot as plt
from megatron.core.transformer.moe.moe_utils import topk_softmax_with_capacity
import numpy as np
import pandas as pd
import pandas as pd
import random
import glob
from collections import defaultdict
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import sys
import random
sys.path.append("../../../../")  # 把项目根目录加入搜索路径
from utils.trace import  *
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
def get_prefilling_layer_predict(args, num_layer,topk,step ):
    
    request_set = set(input_data["request"].unique())
    avg_input_similarities = []
    # --------- 输入相似度 ---------
    for i in range(1,num_layer):
        layer = i
        next_layer = layer + step
        if next_layer > num_layer:
            break
        similarity_list = []
        for request in request_set:
            input_tensor = get_tensor(input_data, layer, request, 0 )
            next_input_tensor =  get_tensor(input_data, next_layer, request, 0 )
            similarity_list +=  get_similarity( input_tensor,next_input_tensor ).tolist()  
        mean_sim = sum(similarity_list) / len(similarity_list)
        print(f"[INPUT SIM] Layer {layer} → {next_layer}: {mean_sim:.4f}")
        avg_input_similarities.append(mean_sim)
    # --------- 输出相似度 & token routing accuracy ---------
    avg_output_similarities = []
    token_expert_predict_accu = []
    total_correct_list = []
    total_expert_list = []
    for layer in range(1, num_layer):
        next_layer = layer + step
        if next_layer > num_layer:
            break
        out_similarity_list = []
        layer_predict_accu = []
        for request in request_set:
            # === predict next layer output ===
            pre_layer_input = get_tensor(input_data, layer, request, 0 )
            
            pred_nex_layer_output = gating( pre_layer_input, load_trained_gate( gate_path, layer,next_layer, step),args )
            nex_layer_output = get_tensor(logits_data, next_layer, request, 0 )
            out_similarity_list +=  get_similarity(pred_nex_layer_output, nex_layer_output ).tolist()  
            ######################################
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
            total_correct_list.append(total_correct.float().item())
            total_expert_list.append(total_expert.float().item())
            avg_acc = total_correct.float() / total_expert.float()
            layer_predict_accu.append(avg_acc.item())
        mean_out_sim = sum(out_similarity_list) / len(out_similarity_list) if out_similarity_list else 0.0
        mean_acc = sum(layer_predict_accu) / len(layer_predict_accu) if layer_predict_accu else 0.0
        print(f"[OUTPUT SIM] Layer {layer} → {next_layer}: {mean_out_sim:.4f}")
        print(f"[ROUTING ACC] Layer {layer} → {next_layer}: {mean_acc:.4f}")
        avg_output_similarities.append(mean_out_sim)
        token_expert_predict_accu.append(mean_acc)
        # token_expert_predict_accu.append(sum(total_correct_list) / sum(total_expert_list)  )
        # print(f"[ROUTING ACC] Layer {layer} → {next_layer}: {token_expert_predict_accu[-1]:.4f}")
        
    return {
        "input_similarities" : avg_input_similarities,
        "output_similarities": avg_output_similarities,
        "token_expert_predict_accuracy":token_expert_predict_accu
        } 
def get_decoding_layer_predict(args,num_layer, topk, step):
    request_set = set(input_data["request"].unique())
    avg_input_similarities = []

    # --------- 输入相似度 ---------
    for i in range(1, num_layer):
        layer = i
        next_layer = layer + step
        if next_layer > num_layer:
            break

        similarity_list = []
        for request in request_set:
            # 获取所有 iter > 0 的 iter 列表
            iters = input_data[
                (input_data["layer"] == layer) &
                (input_data["request"] == request) &
                (input_data["iter"] > 0  )
            ]["iter"].unique()
            for iter_id in iters:
                input_tensor = get_tensor(input_data, layer, request, iter_id)
                next_input_tensor = get_tensor(input_data, next_layer, request, iter_id)
                similarity_list += get_similarity(input_tensor, next_input_tensor).tolist()
        mean_sim = sum(similarity_list) / len(similarity_list) if similarity_list else 0.0
        print(f"[INPUT SIM] Layer {layer} → {next_layer}: {mean_sim:.4f}")
        avg_input_similarities.append(mean_sim)

    # --------- 输出相似度 & token routing accuracy ---------
    avg_output_similarities = []
    token_expert_predict_accu = []

    for layer in range(1, num_layer):
        next_layer = layer + step
        if next_layer > num_layer:
            break

        out_similarity_list = []
        layer_predict_accu = []

        for request in request_set:
            # 获取所有 iter > 0 的 iter 列表
            iters = input_data[
                (input_data["layer"] == layer) &
                (input_data["request"] == request) &
                (input_data["iter"] > 0)
            ]["iter"].unique()

            for iter_id in iters:
                pre_layer_input = get_tensor(input_data, layer, request, iter_id)
                pred_next_layer_output = gating(pre_layer_input, load_trained_gate(gate_path,layer,next_layer, step),args)
                next_layer_output = get_tensor(logits_data, next_layer, request, iter_id)

                out_similarity_list += get_similarity(pred_next_layer_output, next_layer_output).tolist()

                # Routing accuracy
                pred_flat = pred_next_layer_output.view(-1, pred_next_layer_output.shape[-1])
                _, pred_routing_map, _ = topk_softmax_with_capacity(
                    pred_flat,
                    topk=topk,
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

                real_routing_map = get_tensor(map_data, next_layer, request, iter_id)
                correct_map = pred_routing_map & real_routing_map
                total_correct = correct_map.sum()
                total_expert = real_routing_map.sum()
                avg_acc = total_correct.float() / total_expert.float() if total_expert > 0 else torch.tensor(0.0)
                layer_predict_accu.append(avg_acc.item())


        mean_out_sim = sum(out_similarity_list) / len(out_similarity_list) if out_similarity_list else 0.0
        mean_acc = sum(layer_predict_accu) / len(layer_predict_accu) if layer_predict_accu else 0.0

        print(f"[OUTPUT SIM] Layer {layer} → {next_layer}: {mean_out_sim:.4f}")
        print(f"[ROUTING ACC] Layer {layer} → {next_layer}: {mean_acc:.4f}")
        avg_output_similarities.append(mean_out_sim)
        token_expert_predict_accu.append(mean_acc)

    return {
        "input_similarities": avg_input_similarities,
        "output_similarities": avg_output_similarities,
        "token_expert_predict_accuracy": token_expert_predict_accu
    }


def get_all_layer_predict(args,num_layer, topk, step):
    request_set = set(input_data["request"].unique())
    avg_input_similarities = []

    # --------- 输入相似度 ---------
    for i in range(1, num_layer):
        layer = i
        next_layer = layer + step
        if next_layer > num_layer:
            break

        similarity_list = []
        for request in request_set:
            # 获取所有 iter > 0 的 iter 列表
            iters = input_data[
                (input_data["layer"] == layer) &
                (input_data["request"] == request) &
                (input_data["iter"] >= 0  )
            ]["iter"].unique()
            for iter_id in iters:
                input_tensor = get_tensor(input_data, layer, request, iter_id)
                next_input_tensor = get_tensor(input_data, next_layer, request, iter_id)
                similarity_list += get_similarity(input_tensor, next_input_tensor).tolist()
        mean_sim = sum(similarity_list) / len(similarity_list) if similarity_list else 0.0
        print(f"[INPUT SIM] Layer {layer} → {next_layer}: {mean_sim:.4f}")
        avg_input_similarities.append(mean_sim)

    # --------- 输出相似度 & token routing accuracy ---------
    avg_output_similarities = []
    token_expert_predict_accu = []

    for layer in range(1, num_layer):
        next_layer = layer + step
        if next_layer > num_layer:
            break

        out_similarity_list = []
        layer_predict_accu = []

        for request in request_set:
            # 获取所有 iter > 0 的 iter 列表
            iters = input_data[
                (input_data["layer"] == layer) &
                (input_data["request"] == request) &
                (input_data["iter"] >= 0)
            ]["iter"].unique()

            for iter_id in iters:
                pre_layer_input = get_tensor(input_data, layer, request, iter_id)
                pred_next_layer_output = gating(pre_layer_input, load_trained_gate(gate_path,layer,next_layer, step),args)
                next_layer_output = get_tensor(logits_data, next_layer, request, iter_id)

                out_similarity_list += get_similarity(pred_next_layer_output, next_layer_output).tolist()

                # Routing accuracy
                pred_flat = pred_next_layer_output.view(-1, pred_next_layer_output.shape[-1])
                _, pred_routing_map, _ = topk_softmax_with_capacity(
                    pred_flat,
                    topk=topk,
                    
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

                real_routing_map = get_tensor(map_data, next_layer, request, iter_id)
                correct_map = pred_routing_map & real_routing_map
                total_correct = correct_map.sum()
                total_expert = real_routing_map.sum()
                avg_acc = total_correct.float() / total_expert.float() if total_expert > 0 else torch.tensor(0.0)
                layer_predict_accu.append(avg_acc.item())


        mean_out_sim = sum(out_similarity_list) / len(out_similarity_list) if out_similarity_list else 0.0
        mean_acc = sum(layer_predict_accu) / len(layer_predict_accu) if layer_predict_accu else 0.0

        print(f"[OUTPUT SIM] Layer {layer} → {next_layer}: {mean_out_sim:.4f}")
        print(f"[ROUTING ACC] Layer {layer} → {next_layer}: {mean_acc:.4f}")
        avg_output_similarities.append(mean_out_sim)
        token_expert_predict_accu.append(mean_acc)

    return {
        "input_similarities": avg_input_similarities,
        "output_similarities": avg_output_similarities,
        "token_expert_predict_accuracy": token_expert_predict_accu
    }



# save_dir = "./saved_iter_20"
save_dir ="/home/download/models/mg_core/Qwen1.5-MoE-A2.7B-Chat/200_1/"
gate_path = "/home/CodeSpace/Megatron-LM-core_v0.12.0/Qwen/Predict/train/trained_gate"

input_data, num_layers,num_requests =  read_trace(save_dir, "*input**.pt")
logits_data  ,_ ,_ = read_trace(save_dir, "*logits**.pt")
map_data ,_ ,_ = read_trace(save_dir, "*routing_map**.pt")


request_set = set(input_data["request"].unique())
request_list = sorted(list(input_data["request"].unique()))
random.seed(42)  # 保证可复现
random.shuffle(request_list)
split_idx = int(len(request_list) * 0.9)
train_requests = set(request_list[:split_idx])
test_requests = set(request_list[split_idx:])
print(f"Total: {len(request_list)}, Train: {len(train_requests)}, Test: {len(test_requests)}")

input_data = input_data[input_data["request"].isin(test_requests)]
logits_data =  logits_data[logits_data["request"].isin(test_requests)]
map_data =  map_data[map_data["request"].isin(test_requests)]
print(f"Filtered input_data: {len(input_data)} rows")
print(f"Filtered logits_data: {len(logits_data)} rows")
print(f"Filtered map_data: {len(map_data)} rows")


expert_budget_res = dict()
for topk in [4,6,8,10]:
    print("topk",topk)  
    result =  dict()
    for step in range(1,5):
        data= get_prefilling_layer_predict(args,num_layers,topk=topk,step=step)
        result [step] = data
    expert_budget_res[topk] =  result
    plot(result, f"prefilling_expert_budget{topk}.pdf","Prefilling")
plot_expert_budget(expert_budget_res,"expert_acc.pdf" )
plot_expert_budget_by_step(expert_budget_res,  "token_expert_accuracy")

# save_similarity(result,"prefilling.csv", "prefilling")

# result =  dict()
# for step in range(1,5):
#     data= get_decoding_layer_predict(num_layers,topk=topk,step=step)
#     result [step] = data
# plot(result, "decoding.pdf", "Decoding")
# save_similarity(result,"decoding.csv", "decoding")



# result =  dict()
# for step in range(1,5):
#     data= get_all_layer_predict(num_layers,topk=topk,step=step)
#     result [step] = data
# plot(result, "all.pdf","Prefilling+Decoding")
# save_similarity(result,"all.csv", "all")

