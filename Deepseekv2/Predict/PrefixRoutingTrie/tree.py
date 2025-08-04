import sys
sys.path.append("../../../")  # 把项目根目录加入搜索路径
from utils.trace import  *
from argparse import Namespace
import random
from collections import defaultdict, Counter
from tqdm import tqdm
import time
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

 

topk_per_token  =  get_tensor( map_data, 2, 0,0).sum(dim=1)
if not torch.all(topk_per_token == topk_per_token[0]):
    raise ValueError(f"Inconsistent topk per token: {topk_per_token.tolist()}")
topk = topk_per_token[0].item()
print("topk",topk)
request_set = set(input_data["request"].unique())


request_set = set(input_data["request"].unique())
request_list = sorted(list(input_data["request"].unique()))
random.seed(42)  # 保证可复现
random.shuffle(request_list)
split_idx = int(len(request_list) * 0.9)
train_requests = set(request_list[:split_idx])
test_requests = set(request_list[split_idx:])
print(f"Total: {len(request_list)}, Train: {len(train_requests)}, Test: {len(test_requests)}")
from collections import defaultdict, Counter

train_requests_list = sorted(list(train_requests))
test_requests_list = sorted(list(test_requests))
_,E =  get_tensor(map_data,2, 0, 0).shape 
print(E)

def read_expert_trace(requests_list):
    print("Read expert trace")
    all_trace = []
    for req in requests_list :
        T,E =get_tensor(map_data, 2, req, 0).shape 
        trace = [[] for _ in range(T)]  # trace[token] = list of expert sets (per layer)
        for layer in range(2,  27+1):
            layer_map = get_tensor(map_data, layer, req, 0)  # shape [T, E]
            for token in range(T):
                experts = torch.nonzero(layer_map[token], as_tuple=False).squeeze(1)
                trace[token].append(experts.tolist())
        # print(f"T={len(trace)}, Layer_num={len(trace[0])}, E={len(trace[0][0])}")
        all_trace += trace
    return all_trace
def make_perfix_tree(profile_trace, L):
    prefix_tree = defaultdict(Counter)
    for token_trace in profile_trace:
        if len(token_trace) <= L:
            continue  # 长度不足，无法预测 L+1 层
        prefix = tuple(tuple(sorted(e)) for e in token_trace[:L])  # 每层 expert 集合 hash 成 tuple
        next_experts = token_trace[L]  # 第 L+1 层的 expert list
        for e in next_experts:
            prefix_tree[prefix][e] += 1
    return prefix_tree

def jaccard_sim(set1, set2):
    set1, set2 = set(set1), set(set2)
    if not set1 and not set2:
        return 1.0
    return len(set1 & set2) / len(set1 | set2)
def prefix_similarity(p1, p2):
    # p1, p2 are both tuples of tuples
    return sum(jaccard_sim(a, b) for a, b in zip(p1, p2)) / len(p1)
def predict_next_expert(prefix_tree, prefix, topk=6):
    key = tuple(tuple(sorted(e)) for e in prefix)
    if key in prefix_tree:
        counter = prefix_tree[key]
    else:
        best_key = None
        best_sim = -1
        for candidate_key in prefix_tree:
            if len(candidate_key) != len(key):
                continue
            sim = prefix_similarity(key, candidate_key)
            if sim > best_sim:
                best_sim = sim
                best_key = candidate_key
        if best_key is None:
            return None  # no valid prefix found at all
        print(f"Using fallback prefix with similarity {best_sim:.3f}")
        counter = prefix_tree[best_key]
    total = sum(counter.values())
    sorted_probs = sorted(counter.items(), key=lambda x: -x[1])
    res =  [(e, c / total) for e, c in sorted_probs[:topk]]
    expert = [i[0] for i in res]
    probs = [i[1] for i in res]
    return expert, probs
def evaluate_prediction(true_set, pred_set):
    true_set = set(true_set)
    pred_set = set (pred_set)
    hit = len(true_set & pred_set)
    precision = hit / len(pred_set)
    recall = hit / len(true_set)
    jaccard = hit / len(true_set | pred_set)
    return{
        "precision":precision,
        "recall":recall,
        "jaccard":jaccard,
        } 


profile_trace = read_expert_trace (train_requests_list )
print(len(profile_trace))
test_trace = read_expert_trace (test_requests_list )
print(len(test_trace))
summary = []


num_layer = len(profile_trace[0])
for L in range(7,num_layer):
    if L+1 >= num_layer:
        break
    prefix_tree = make_perfix_tree(profile_trace, L)
    print(f"L={L}, prefix_tree size = {len(prefix_tree)}")
    record = []
    start_time = time.time()
    for test_token in tqdm(test_trace , desc="Evaluating"):
        
        prefix  = test_token[:L]  # 用一个 token 的前缀做测试
        true_expert =  test_token[ L]
        prediction = predict_next_expert(prefix_tree, prefix, topk=6)
        print(f"True:      {sorted(true_expert)}")
        print(f"Predicted: {sorted(prediction[0])}")
        score = evaluate_prediction(true_expert  , prediction[0] )
        score["L"] = L
        record.append(score)
    end_time  =  time.time()
    elapsed = end_time - start_time
    df = pd.DataFrame(record)
    csv_path = f"prefix_routing_L{L}_eval.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved evaluation results to: {csv_path}")
    
    avg_precision = df["precision"].mean() if not df.empty else 0
    avg_recall = df["recall"].mean() if not df.empty else 0
    avg_jaccard = df["jaccard"].mean() if not df.empty else 0
    summary.append({
        "L": L,
        "prefix_tree_size": len(prefix_tree),
        "num_samples": len(df),
        "avg_precision": avg_precision,
        "avg_recall": avg_recall,
        "avg_jaccard": avg_jaccard,
        "time_seconds": elapsed
    })
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv("prefix_routing_summary.csv", index=False)
    print("Saved overall summary to: prefix_routing_summary.csv")
 