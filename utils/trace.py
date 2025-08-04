import re
import ast
import torch.nn.functional as F
import os
 
import matplotlib.pyplot as plt
from megatron.core.transformer.moe.moe_utils import topk_softmax_with_capacity
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch.nn as nn
import glob
from collections import defaultdict
import torch
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_trained_gate(path,layer, next_layer, step ):
    filename = f"layer_{layer}_next_layer{next_layer}_step{step}.pt"
    path = os.path.join(path, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    weight = torch.load(path, map_location=device)
    return weight["gate"]


def load_gate(path):
    print("load gate", path)
    # load gate for each layer
    mg_model_state = torch.load(path,map_location="cpu", weights_only=False)["model"]
    result = {}
    for key, value in mg_model_state.items():
        if "router" in key:
            parts = key.split(".")
            layer_idx = int(parts[2])
            # print(key, layer_idx,value.shape, value.dtype)
            # NOTE +1 for megatron layer, in mg-core weight, 0-index
            result[layer_idx + 1] = value.to(device)
    # note 
    print(f"Load {len(result)} gate" , result.keys())
    return result
def gating(input, gate_weight,args ):
    
    router_dtype = input.dtype
    if args.moe_router_dtype == 'fp32':
        router_dtype = torch.float32
    elif args.moe_router_dtype == 'fp64':
        router_dtype = torch.float64
    else:
        raise ValueError(f"Unsupported moe_router_dtype: {args.moe_router_dtype}")
    
    logits = torch.nn.functional.linear(input.to(router_dtype), gate_weight.to(router_dtype))
    return logits
    
 
def read_trace(path, str ):
    pattern = os.path.join(path,str )
    print(f"\n[INFO] Scanning: {pattern}")
    input_files = glob.glob(pattern)
    data = defaultdict(dict)
    print(f"[INFO] Found {len(input_files)} matching files.")
    records = []
    layer_set = set()
    request_set =  set()
    for file in input_files:
        filename = os.path.basename(file)
        match = re.search(r"layer_(\d+)_request_(\d+)_iter_(\d+)\.pt$", filename)
        if match:
            layer_id = int(match.group(1))
            request_id = int(match.group(2))
            iter_id = int(match.group(3))
            if layer_id==0:
                layer_id = 1
            layer_set.add(layer_id)
            request_set.add(request_id)
            records.append({
                "layer": layer_id,
                "request": request_id,
                "iter": iter_id,
                "path": file
            })
        else:
            print(f"[WARNING] No match found for file: {file}")
    df = pd.DataFrame(records)
    num_layers = len(layer_set)
    num_requests = len(request_set)  
    print(f"[SUMMARY] Total layers: {num_layers} -> {sorted(layer_set)}")
    print(f"[SUMMARY] Total requests per layer: {num_requests} -> {sorted(request_set)}")
    return df,num_layers,num_requests
def get_similarity(data1,data2):
    assert data1.shape == data2.shape, f"Shape mismatch: {data1.shape} vs {data2.shape}"
    hidden_size = data1.shape[-1]
    data1 = data1.view(-1, hidden_size)
    data2 = data2.view(-1, hidden_size)
    
    similarity = F.cosine_similarity(data1, data2, dim=1)
    return similarity
def get_tensor(data, layer, request, iter):
    # 查找匹配的路径
    matched = data[
        (data["layer"] == layer) &
        (data["request"] == request) &
        (data["iter"] == iter)
    ]["path"]
    
    # 检查是否匹配到路径
    if matched.empty:
        raise ValueError(f"No tensor found for layer={layer}, request={request}, iter={iter}")
    # print(" ",matched.values[0])
    # 加载 tensor
    tensor = torch.load(matched.values[0], map_location='cpu')
    return tensor.to(device)
def get_prefilling_layer_predict(args,input_data, logits_data,
                                 map_data,all_gate,num_layer,topk,step, ):
    
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
            avg_acc = total_correct.float() / total_expert.float()
            layer_predict_accu.append(avg_acc.item())
        mean_out_sim = sum(out_similarity_list) / len(out_similarity_list) if out_similarity_list else 0.0
        mean_acc = sum(layer_predict_accu) / len(layer_predict_accu) if layer_predict_accu else 0.0
        print(f"[OUTPUT SIM] Layer {layer} → {next_layer}: {mean_out_sim:.4f}")
        print(f"[ROUTING ACC] Layer {layer} → {next_layer}: {mean_acc:.4f}")
        avg_output_similarities.append(mean_out_sim)
        token_expert_predict_accu.append(mean_acc)
        
    return {
        "input_similarities" : avg_input_similarities,
        "output_similarities": avg_output_similarities,
        "token_expert_predict_accuracy":token_expert_predict_accu
        } 
def get_decoding_layer_predict(args,input_data, logits_data, map_data,all_gate,num_layer, topk, step):
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
                pred_next_layer_output = gating(pre_layer_input, all_gate[next_layer],args)
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


def get_all_layer_predict(args,input_data, logits_data, map_data,all_gate,num_layer, topk, step):
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
                pre_layer_input =  get_tensor(input_data, layer, request, iter_id )
                pred_next_layer_output = gating(pre_layer_input, all_gate[next_layer],args)
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
 
    
def plot_expert_budget(expert_budget_res, path):
        
    topks = sorted(expert_budget_res.keys())
    num_topks = len(topks)
    # 子图排列方式
    cols = 2
    rows = math.ceil(num_topks / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5), squeeze=False)
    # fig.suptitle("Token-Expert Predict Accuracy across Steps", fontsize=22)
    for i, topk in enumerate(topks):
        ax = axes[i // cols][i % cols]
        data =   expert_budget_res[topk] 
        steps = sorted(data.keys())
        for step in steps:
            layers = list(range(1, 1 + len(data[step]["token_expert_predict_accuracy"] )))
            ax.plot(data[step]["token_expert_predict_accuracy"] , marker='s', label=f'step={step}')

        ax.set_title(f"Expert Budget = {topk}", fontsize=14)
        ax.set_xlabel("Layer Index", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(True)
        

    # 移除多余子图
    for j in range(i + 1, rows * cols):
        fig.delaxes(axes[j // cols][j % cols])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(path)
def plot(result, path,label):
    steps = sorted(result.keys())
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    for step in steps:
        # 对于 step=x，相似度是 layer i vs i+x，因此 x 不能超过 num_layer-x
        layers = list(range(1, 1 + len(result[step]["input_similarities"])))
        axs[0].plot(layers, result[step]["input_similarities"], marker='o', label=f'step={step}')
    axs[0].set_title(f"{label } Gating Input Similarity (Layer i vs i + step)")
    axs[0].set_xlabel("Layer Index")
    axs[0].set_ylabel("Cosine Similarity")
    axs[0].set_ylim(0, 1)  # 强制 y 轴范围为 [0, 1]
    axs[0].grid(True)
    axs[0].legend()
    for step in steps:
        layers = list(range(1, 1 + len(result[step]["output_similarities"])))
        axs[1].plot(layers, result[step]["output_similarities"], marker='x', label=f'step={step}')
    axs[1].set_title(f"{label } Predicted vs. Actual Gating Output Similarity")
    axs[1].set_xlabel("Layer Index")
    axs[1].set_ylabel("Cosine Similarity")
    axs[1].set_ylim(0, 1)  # 强制 y 轴范围为 [0, 1]
    axs[1].grid(True)
    axs[1].legend()
    for step in steps:
        layers = list(range(1, 1 + len(result[step]["token_expert_predict_accuracy"])))
        axs[2].plot(layers, result[step]["token_expert_predict_accuracy"], marker='s', label=f'step={step}')
    axs[2].set_title(f"{label } Expert Selection Accuracy (Predicted at Layer i)")
    axs[2].set_xlabel("Layer Index")
    axs[2].set_ylabel("Accuracy")
    axs[2].grid(True)
    axs[2].set_ylim(0, 1)  # 强制 y 轴范围为 [0, 1]
    axs[2].legend()

    plt.tight_layout()
    plt.savefig(path)

def save_similarity(result_dict, save_path, label):
    rows = []
    for step, data in result_dict.items():
        rows.append({
            "type": label,
            "step": step,
            "input_similarities": data["input_similarities"],
            "output_similarities": data["output_similarities"],
            "token_expert_predict_accuracy": data["token_expert_predict_accuracy"]
        })

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"[INFO] Saved to {save_path}")
class TrainableRouter(nn.Module):
    def __init__(self, gate_weight,args):
        super().__init__()
        # 将 gate_weight 转为可训练参数（注意：必须是 float）
        self.gate = nn.Parameter(gate_weight.float())
        self.args = args
    def forward(self, x,  ):
        router_dtype = x.dtype
        if self.args.moe_router_dtype == 'fp32':
            router_dtype = torch.float32
        elif self.args.moe_router_dtype == 'fp64':
            router_dtype = torch.float64
        else:
            raise ValueError(f"Unsupported moe_router_dtype: {self.args.moe_router_dtype}")
        return F.linear(x.to(router_dtype), self.gate.to(router_dtype))

def eval(test_requests, model, topk,layer, next_layer,args,input_data,map_data):
    model.eval() 
    acc_list = []
    for request in test_requests:
        pre_layer_input = get_tensor(input_data, layer, request, 0 ).to(device).float()
        with torch.no_grad():
            pred = model(pre_layer_input)
            pred =  pred.view(-1,  pred.shape[-1])
            _, pred_routing_map, _ = topk_softmax_with_capacity(
                pred,
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
            real_routing_map = get_tensor(map_data,next_layer,request,0).to(device)
            correct_map = pred_routing_map & real_routing_map 
            total_correct  = correct_map.sum()
            total_expert = real_routing_map.sum()
            avg_acc = total_correct.float() / total_expert.float()
            acc_list.append(avg_acc.item())
    mean_acc = sum(acc_list) / len(acc_list)
    print(f"[Eval] Mean Accuracy on {len(test_requests)} requests: {mean_acc:.4f}")
    model.train()  # 恢复训练模式
    return mean_acc

def plot_expert_budget_by_step(expert_budget_res, path_prefix):
    budgets = sorted(expert_budget_res.keys())
    steps = sorted(list(expert_budget_res[budgets[0]].keys()))

    cols = 2
    rows = math.ceil(len(steps) / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5), squeeze=False)

    for idx, step in enumerate(steps):
        ax = axes[idx // cols][idx % cols]
        for budget in budgets:
            acc_list = expert_budget_res[budget][step]["token_expert_predict_accuracy"]
            layers = list(range(1, 1 + len(acc_list)))
            ax.plot(layers, acc_list, marker='o', label=f'Budget={budget}')
        
        ax.set_title(f"Step = {step}", fontsize=14)
        ax.set_xlabel("Layer Index", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_ylim(0, 1)
        ax.grid(True)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.legend(fontsize=9, loc='lower right')

    plt.tight_layout()
    plt.savefig(f"{path_prefix}_by_step.pdf")
    plt.close()