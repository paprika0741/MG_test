import re
import matplotlib.pyplot as plt
import numpy as np
import re
from collections import Counter
import glob
import statistics
import pandas as pd
import math
def parse_shape_counts(log_file, warm_up=10):
    
    pattern = re.compile(r'RANK\[0\] layer 1 : hidden_states\.shape\s*=\s*torch\.Size\(\[([\d]+),')
    shape_counts = []
    matched = 0  # count how many matches seen so far

    with open(log_file, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                matched += 1
                if matched <= warm_up:
                    continue  # skip first `warm_up` matches
                dim0 = int(match.group(1))
                shape_counts.append(dim0)

    counter = Counter(shape_counts)
    avg = np.mean(shape_counts) if shape_counts else 0
    print(f"[Summary for {log_file}]")
    print(f"  ▸ Warm-up skipped     : {warm_up}")
    print(f"  ▸ Total valid entries : {len(shape_counts)}")
    print(f"  ▸ Average Prompt Length: {avg:.2f}")
    print("  ▸ Distribution of Prompt Lengths:")

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.hist(shape_counts, bins=range(min(shape_counts), max(shape_counts)+1), edgecolor='black', align='left')
    plt.title("Distribution of Prompts Length")
    plt.xlabel("Prompts Length")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("prompt_len.png")
    return counter

# Example usage:
# parse_shape_counts("moe_layer_log.txt")

    
def parser_logs(file,warm_up=10, layer_id = 1 ):
    pattern = re.compile(rf'RANK\[0\] moe layer {layer_id} elapsed ([0-9.]+) ms')
    elapsed_times = []
    # Match pattern: idealX_skewY_eplbZ
    match = re.search(r"ideal(\d+)_skew(\d+)_eplb(\d+)", file)
    if match:
        ideal, skew, eplb = map(int, match.groups())
        print(f"ideal={ideal}, skew={skew}, eplb={eplb}")
    else:
        raise ValueError(f"Filename {file} does not match expected pattern.")
    data_tag = "Artificial data" if skew else "Real data"
    method_tag = "EPLB" if eplb else ("IDEAL" if ideal else "Megatron")
    tag = f"{data_tag}+{method_tag}"
    
    with open(file, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                elapsed_time = float(match.group(1))
                elapsed_times.append(elapsed_time)
    res = elapsed_times[warm_up:]
    print(f"[Summary for {file}] {tag} layer {layer_id}")
    print(f"  ▸ Total entries      : {len(elapsed_times)}")
    print(f"  ▸ Warm-up skipped    : {warm_up}")
    print(f"  ▸ Effective entries  : {len(res)}")
    return res,tag
def plot_all_layers_latency_cdf(df, total_layers, real_data=True):
    """
    将所有 layer 的 CDF 曲线画在同一张图中（多个子图）。

    参数:
        df (DataFrame): 包含 columns ["layer_id", "tag", "times"] 的 DataFrame。
        total_layers (int): 层数。
    """
    rows = math.ceil(total_layers / 4)
    fig, axs = plt.subplots(rows, 4, figsize=(20, 3 * rows))
    axs = axs.flatten()

    for i in range(total_layers):
        layer_id = i + 1
        ax = axs[i]
        filtered_df = df[df["layer_id"] == layer_id ]
        print(filtered_df)
        for _, row in filtered_df.iterrows():
            tag = row["tag"]
            times = row["times"]
            if real_data and "Real data" in tag:
                sorted_times = np.sort(times)
                cdf = np.linspace(0, 1, len(sorted_times))
                ax.plot(sorted_times, cdf, label=tag)
            if not real_data and "Artificial data" in tag:
                sorted_times = np.sort(times)
                cdf = np.linspace(0, 1, len(sorted_times))
                ax.plot(sorted_times, cdf, label=tag)
                
                
        
        ax.set_title(f"Layer {layer_id}")
        ax.set_xlabel("Latency (ms)")
        ax.set_ylabel("CDF")
        ax.grid(True)
        ax.legend(fontsize=8)

    # 清空多余子图
    for j in range(total_layers, len(axs)):
        fig.delaxes(axs[j])

    plt.suptitle("Latency CDF per MOE Layer (RANK[0])", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if real_data:
        plt.savefig("all_layers_latency_cdf_Real_data.pdf")
    else:
        plt.savefig("all_layers_latency_cdf_Artificial_data.pdf")
    plt.close()

file_list = glob.glob("*.log")
print("Found .log files:", file_list)
warm_up_sample = 20
parse_shape_counts(file_list[0], warm_up_sample)
layer_num = 8
data = []
for layer_id in range(1,layer_num + 1):
    for file in file_list:
        times,tag = parser_logs(file, warm_up_sample, layer_id )
        data.append({
                "layer_id": layer_id,
                "tag": tag,
                "times": times,
                "sample_num": len(times)
            })
df = pd.DataFrame(data)
plot_all_layers_latency_cdf(df, total_layers=layer_num, real_data=True)
plot_all_layers_latency_cdf(df, total_layers=layer_num, real_data=False)
for layer_id in range(1,layer_num + 1):
    filtered_df = df[df["layer_id"] == layer_id ]
    # print(filtered_df)
    print("======================================")
    for data_type in ["Real data", "Artificial data"]:
        megatron_row = filtered_df[filtered_df["tag"] == f"{data_type}+Megatron"]
        ideal_row = filtered_df[filtered_df["tag"] == f"{data_type}+IDEAL"]
        eplb_row = filtered_df[filtered_df["tag"] == f"{data_type}+EPLB"]
        
        if not megatron_row.empty and not ideal_row.empty:
            avg_megatron = np.mean(megatron_row["times"].values[0])
            avg_ideal = np.mean(ideal_row["times"].values[0])
            speedup_ideal = avg_megatron / avg_ideal
            print(f"[Layer {layer_id}] {data_type} IDEAL over Megatron: {speedup_ideal:.4f}x")
        if not megatron_row.empty and not eplb_row.empty:
            avg_megatron = np.mean(megatron_row["times"].values[0])
            avg_eplb = np.mean(eplb_row["times"].values[0])
            speedup_eplb = avg_megatron / avg_eplb
            print(f"[Layer {layer_id}] {data_type} EPLB  over Megatron: {speedup_eplb:.4f}x")
    
    
    
    
    
    
    
    
    
    # print("Real data")
    # print("->", "ideal over megatron")
    # megatron  = res_dict["Real data+Megatron"]
    # ideal  = res_dict["Real data+IDEAL"]
    # speedups = [m / i for m, i in zip(megatron, ideal)]
    # avg_speedup = statistics.mean(speedups)
    # print(f"Average speedup: {avg_speedup:.4f}")


    # print("->", "eplb over megatron")
    # megatron  = res_dict["Real data+Megatron"]
    # eplb  = res_dict["Real data+EPLB"]
    # speedups = [m / i for m, i in zip(megatron, eplb)]
    # avg_speedup = statistics.mean(speedups)
    # print(f"Average speedup: {avg_speedup:.4f}")

    # print("Artificial data")
    # print("->", "ideal over megatron")
    # megatron  = res_dict["Artificial data+Megatron"]
    # ideal  = res_dict["Real data+IDEAL"]
    # speedups = [m / i for m, i in zip(megatron, ideal)]
    # avg_speedup = statistics.mean(speedups)
    # print(f"Average speedup: {avg_speedup:.4f}")


    # print("->", "eplb over megatron")
    # megatron  = res_dict["Artificial data+Megatron"]
    # eplb  = res_dict["Artificial data+EPLB"]
    # speedups = [m / i for m, i in zip(megatron, eplb)]
    # avg_speedup = statistics.mean(speedups)
    # print(f"Average speedup: {avg_speedup:.4f}")
