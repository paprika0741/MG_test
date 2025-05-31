import re
import matplotlib.pyplot as plt
import numpy as np
import re
from collections import Counter
import glob
import statistics

def parse_shape_counts(log_file, warm_up=10):
    pattern = re.compile(r'RANK\[0\].*?hidden_states\.shape\s*=\s*torch\.Size\(\[([\d]+),')
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

    
def parser_logs(file,warm_up=10):
    pattern = re.compile(r'RANK\[0\] moe layer elapsed ([0-9.]+) ms')
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
    print(f"[Summary for {file}] {tag}")
    print(f"  ▸ Total entries      : {len(elapsed_times)}")
    print(f"  ▸ Warm-up skipped    : {warm_up}")
    print(f"  ▸ Effective entries  : {len(res)}")
    return res,tag
def plot_latency_cdf(res):
    """
    绘制每组配置的 latency CDF

    参数:
        res: List of (tag, times), 每组是一个配置下的耗时列表
    """
    plt.figure(figsize=(8, 6))
    for tag, times in res:
        sorted_times = np.sort(times)
        cdf = np.linspace(0, 1, len(sorted_times))
        plt.plot(sorted_times, cdf, label=tag)
    plt.xlabel("Latency (ms)")
    plt.ylabel("CDF")
    plt.title("MOE Layer Latency CDF (RANK[0])")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("result.png")
file_list = glob.glob("*.log")
print("Found .log files:", file_list)
parse_shape_counts(file_list[0], 10)
res = []
res_dict = {}
for file in file_list:
    times,tag = parser_logs(file, 10)
    res_dict[tag] = times
    print(len(times),tag )
    res.append((tag,times))
plot_latency_cdf(res)

print("Real data")
print("->", "ideal over megatron")
megatron  = res_dict["Real data+Megatron"]
ideal  = res_dict["Real data+IDEAL"]
speedups = [m / i for m, i in zip(megatron, ideal)]
avg_speedup = statistics.mean(speedups)
print(f"Average speedup: {avg_speedup:.4f}")


print("->", "eplb over megatron")
megatron  = res_dict["Real data+Megatron"]
eplb  = res_dict["Real data+EPLB"]
speedups = [m / i for m, i in zip(megatron, eplb)]
avg_speedup = statistics.mean(speedups)
print(f"Average speedup: {avg_speedup:.4f}")

print("Artificial data")
print("->", "ideal over megatron")
megatron  = res_dict["Artificial data+Megatron"]
ideal  = res_dict["Real data+IDEAL"]
speedups = [m / i for m, i in zip(megatron, ideal)]
avg_speedup = statistics.mean(speedups)
print(f"Average speedup: {avg_speedup:.4f}")


print("->", "eplb over megatron")
megatron  = res_dict["Artificial data+Megatron"]
eplb  = res_dict["Artificial data+EPLB"]
speedups = [m / i for m, i in zip(megatron, eplb)]
avg_speedup = statistics.mean(speedups)
print(f"Average speedup: {avg_speedup:.4f}")
