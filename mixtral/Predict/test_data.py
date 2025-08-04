import sys
sys.path.append("../../")  # 把项目根目录加入搜索路径
from utils.trace import  *
save_dir = "/home/download/models/mg_core/Mixtral-8x7B-v0.1/200_1/"
input_data, num_layers,num_requests =  read_trace(save_dir, "*input**.pt")
logits_data  ,_ ,_ = read_trace(save_dir, "*logits**.pt")
map_data ,_ ,_ = read_trace(save_dir, "*routing_map**.pt")

path = "/home/download/models/mg_core/Mixtral-8x7B-v0.1/mcore-TP1PP1EP4/iter_0000001/mp_rank_00_000/model_optim_rng.pt"
all_gate = load_gate(path)
print(input_data.head())
 
topk_per_token  =  get_tensor( map_data, 2, 0,0).sum(dim=1)
if not torch.all(topk_per_token == topk_per_token[0]):
    raise ValueError(f"Inconsistent topk per token: {topk_per_token.tolist()}")
topk = topk_per_token[0].item()
print("topk",topk)

step =  0
for layer in range(1,num_layers+1):
    next_layer = layer + step
    if next_layer > num_layers:
            break
    input_tensor = get_tensor(input_data, layer, 1, 0 )
    output = gating( input_tensor, all_gate[next_layer] )
    output_tensor = get_tensor(logits_data, next_layer, 1, 0 )
    diff = torch.abs(output - output_tensor)
    print("最大差异:", diff.max())
    sim = get_similarity(output,output_tensor)
    print(sim.mean().item())
     