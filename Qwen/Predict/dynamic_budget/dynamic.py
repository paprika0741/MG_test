import sys
import random
import pickle
sys.path.append("../../../")  # 把项目根目录加入搜索路径
from utils.trace import  *
from argparse import Namespace
import matplotlib.pyplot as plt

import time
def static_budget(B, L):
    budgets = {layer: 0 for layer in range(1, L + 1)}
    base = B // L
    remainder = B % L
    budgets = {}
    for layer in range(1, L + 1):
            budgets[layer] = base + (1 if layer <= remainder else 0)
    return budgets
def plot(total_result, method):
    # 1. 排序键值对（确保横轴递增）
    sorted_result = dict(sorted(total_result.items()))
    budgets = list(sorted_result.keys())
    accuracies = list(sorted_result.values())

    # 2. 绘图
    plt.figure(figsize=(6, 4))
    plt.plot(budgets, accuracies, marker='o', markersize=4, linewidth=1)  # 更细的线
    plt.xlabel('Total Expert Budget', fontsize=12)
    plt.ylabel('Average Accuracy (%)', fontsize=12)
    plt.title('Average Accuracy vs. Total Budget', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{method}.pdf")
import matplotlib.pyplot as plt
from itertools import cycle

def compare(result_dict):
    plt.figure(figsize=(6, 4))
    marker_cycle = cycle(['o', 's', 'D', '^', 'v', 'x', '*', 'P', '+'])  # 可自定义 marker 样式
    
    for method, result in result_dict.items():
        sorted_result = dict(sorted(result.items()))
        budgets = list(sorted_result.keys())
        accuracies = list(sorted_result.values())
        marker = next(marker_cycle)
        plt.plot(
            budgets,
            accuracies,
            marker=marker,
            markersize=1,      # 更小的 marker
            linewidth=0.8,     # 更细的线条
            label=method
        )

    plt.xlabel('Total Expert Budget', fontsize=12)
    plt.ylabel('Average Accuracy (%)', fontsize=12)
    plt.title('Average Accuracy vs. Total Budget', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("compare.pdf")
    plt.close()
def dynamic_budget(B, L, K, E, alpha=0.2):
    # Step 1: 计算反比权重（前面层权重更高）
    weights = [math.exp(-alpha * (l - 1)) for l in range(1, L + 1)]
    total_weight = sum(weights)

    # Step 2: 初始分配 + clip 到 [K, E]
    budgets = {}
    for i, w in enumerate(weights, start=1):
        b = int(round(B * w / total_weight))
        budgets[i] = min(max(b, K), E)

    # Step 3: 调整预算以确保 sum(budgets) == B
    current_total = sum(budgets.values())
    diff = B - current_total

    # Step 4: 微调（先加后减，始终保证合法范围）
    while diff != 0:
        updated = False
        if diff > 0:
            for i in range(1, L + 1):
                if budgets[i] < E:
                    budgets[i] += 1
                    diff -= 1
                    updated = True
                    if diff == 0:
                        break
        elif diff < 0:
            for i in range(L, 0, -1):
                if budgets[i] > K:
                    budgets[i] -= 1
                    diff += 1
                    updated = True
                    if diff == 0:
                        break
        if not updated:
            raise ValueError("Unable to adjust budgets to match target total B within [K, E] bounds.")
    
    return budgets



def greedy_budget_allocation(acc_list,   num_layers, total_budget, min_budget=6,E=64):
    budgets = {layer: min_budget for layer in range(1, num_layers + 1)}
    # print("init",budgets )
    used_budget = sum(budgets.values())
    # print("used_budget",used_budget)
    total_accuracy = sum(acc_list[budgets[layer]][layer] for layer in range(1, num_layers + 1))
    # print("init", total_accuracy / num_layers)
    total_accuracy = sum(acc_list[ min_budget][layer] for layer in range(1, num_layers + 1))
    # print("init", total_accuracy / num_layers)
    while used_budget < total_budget:
        best_gain = -1
        best_layer = -1
        for layer in range(1,num_layers + 1) :    
            current_b = budgets[layer]
            next_b = current_b + 1
            if next_b > E:
                continue
            try:
                curr_acc =acc_list[current_b][layer] 
                next_acc =  acc_list[next_b][layer] 
                gain = next_acc - curr_acc
            except KeyError:
                print("error",current_b,next_b,layer)
                continue
            if gain > best_gain:
                best_gain = gain
                best_layer = layer

        if best_layer == -1:
            break   
        budgets[best_layer] += 1
        used_budget += 1

    # 计算最终总准确率
    total_accuracy = sum(acc_list[budgets[layer]][layer] for layer in range(1, num_layers + 1))
    # print("final",   budgets ) 
    # print("final acc",   total_accuracy / num_layers ) 
    return budgets,total_accuracy / num_layers

def dp_budget_allocation(acc_list, L, B, K, E):
    '''
    acc_list[b][l]: 准确率 (float), 第 l 层在 budget b 下的准确率
    L: 总层数，从 1 到 L
    B: 总预算
    K: 每层最小 budget
    E: 每层最大 b
    '''
    dp = [[-1e9] * (B + 1) for _ in range(L + 1)]
    path = [[-1] * (B + 1) for _ in range(L + 1)]  # path[l][b]: 第 l 层用了多少 budget
    dp[0][0] = 0  # base case
    for l in range(1, L + 1):
        for b in range(l*K, B + 1):
            for k in range(K, E + 1):
                if b - k >= 0 and l in acc_list.get(k, {}):
                    prev = dp[l - 1][b - k]
                    curr_acc = acc_list[k][l]
                    if prev + curr_acc > dp[l][b]:
                        dp[l][b] = prev + curr_acc
                        path[l][b] = k
    # 查找最优总准确率和对应 budget
    budgets = {layer: -1 for layer in range(1, L + 1)}
    best_b = max(range(K * L, B + 1), key=lambda x: dp[L][x])
    curr_b = best_b
    max_total_acc = dp[L][best_b]
    for l in range(L, 0, -1):
        budgets[l] = path[l][curr_b]
        curr_b -= budgets[l]
    return budgets, max_total_acc / L
# def dp_budget_allocation(result, )
with open("expert_budget_res.pkl", "rb") as f:
    result = pickle.load(f)

print(result.keys())
budget = list(result.keys())
print("budget", budget)
print( len( result[6][1]["token_expert_predict_accuracy"] ))
layer =  len( result[6][1]["token_expert_predict_accuracy"] )
print(result[6][1]["token_expert_predict_accuracy"][0])

acc_list = dict()
for b in budget:
    acc_list[b] = dict()
    for l in range(len(result[b][1]["token_expert_predict_accuracy"]  )):
        acc_list[b][l+1] = result[b][1]["token_expert_predict_accuracy"]  [l]
total_accuracy = sum(acc_list[ 6 ][l] for l in range(1, layer + 1))  / layer
print("init",total_accuracy )

K = 4 
E = 60
L = 24
print(f"K = {K}, E = {E}, L = {L}")
print("=========================================")
total_result_greedy = dict()
for B in range (L*K, L*E+1):
    start = time.perf_counter()
    budgets, avg_accuracy = greedy_budget_allocation(acc_list, layer, B,  K  ,E)
    total_result_greedy[B] = avg_accuracy
    print("greedy")
    end = time.perf_counter()
    elapsed_ms = (end - start) * 1000
    sum(budgets.values())
    print("Execution time: {:.3f} ms".format(elapsed_ms))
print(total_result_greedy)
plot(total_result_greedy,"greedy")
 
print("=========================================")
print("dp")

# total_result_dp = dict()
# for B in range (L*K, L*E+1):
#     start = time.perf_counter()
#     budgets, avg_accuracy = dp_budget_allocation(acc_list, layer, B,  K  ,E)
#     total_result_dp[B] = avg_accuracy
#     end = time.perf_counter()
#     elapsed_ms = (end - start) * 1000
#     print("Execution time: {:.3f} ms".format(elapsed_ms))
# print(total_result_dp)
# plot(total_result_dp,"dp")
 
 
# static
total_result_static = dict()
for B in range (L*K, L*E+1):
    budgets = static_budget(B,L)
    avg_accuracy = sum(acc_list[budgets[l]][l] for l in range(1, layer + 1)) / layer
    total_result_static[B] = avg_accuracy
plot(total_result_static,"static")

 
total_result_dynamic = dict()
for B in range (L*K, L*E+1):
    budgets = dynamic_budget(B,L,K,E)
    print(budgets)
    avg_accuracy = sum(acc_list[budgets[l]][l] for l in range(1, layer + 1)) / layer
    total_result_dynamic[B] = avg_accuracy
plot(total_result_dynamic,"dynamic")
# compare
compare_result = dict()
compare_result["Average"] = total_result_static
compare_result["Dynamic"] = total_result_dynamic
compare_result["Ours"] = total_result_greedy
# compare_result["dp"] = total_result_dp

compare(compare_result)