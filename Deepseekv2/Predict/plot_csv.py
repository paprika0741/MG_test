import pandas as pd
import matplotlib.pyplot as plt

import ast  # 用于安全地将字符串转为 Python 列表
def plot(result, path,label):
    steps = sorted(result.keys())
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    for step in steps:
        # 对于 step=x，相似度是 layer i vs i+x，因此 x 不能超过 num_layer-x
        layers = list(range(1, 1 + len(result[step]["input_similarities"])))
        axs[0].plot(layers, result[step]["input_similarities"], marker='o', label=f'step={step}')
    axs[0].set_title(f"{label } Gating Input Similarity (Layer i vs i + step)")
    axs[0].set_xlabel("Layer i")
    axs[0].set_ylabel("Cosine Similarity")
    axs[0].set_ylim(0, 1)  # 强制 y 轴范围为 [0, 1]
    axs[0].grid(True)
    axs[0].legend()
    for step in steps:
        layers = list(range(1, 1 + len(result[step]["output_similarities"])))
        axs[1].plot(layers, result[step]["output_similarities"], marker='x', label=f'step={step}')
    axs[1].set_title(f"{label } Predicted vs. Actual Gating Output Similarity")
    axs[1].set_xlabel("Layer i")
    axs[1].set_ylabel("Cosine Similarity")
    axs[1].set_ylim(0, 1)  # 强制 y 轴范围为 [0, 1]
    axs[1].grid(True)
    axs[1].legend()
    for step in steps:
        layers = list(range(1, 1 + len(result[step]["token_expert_predict_accuracy"])))
        axs[2].plot(layers, result[step]["token_expert_predict_accuracy"], marker='s', label=f'step={step}')
    axs[2].set_title(f"{label } Expert Selection Accuracy (Predicted at Layer i)")
    axs[2].set_xlabel("Layer i")
    axs[2].set_ylabel("Accuracy")
    axs[2].grid(True)
    axs[2].set_ylim(0, 1)  # 强制 y 轴范围为 [0, 1]
    axs[2].legend()

    plt.tight_layout()
    plt.savefig(path)
def parse_csv_to_result(csv_path):
    df = pd.read_csv(csv_path)
    result = {}

    for _, row in df.iterrows():
        step = int(row['step'])  # 使用 step 作为 key
        result[step] = {
            "input_similarities": ast.literal_eval(row['input_similarities']),
            "output_similarities": ast.literal_eval(row['output_similarities']),
            "token_expert_predict_accuracy": ast.literal_eval(row['token_expert_predict_accuracy'])
        }

    return result

csv_path = "prefilling.csv"  # 替换为你的 CSV 路径
result = parse_csv_to_result(csv_path)
plot(result, "prefilling.pdf","Prefilling")


csv_path = "decoding.csv"  # 替换为你的 CSV 路径
result = parse_csv_to_result(csv_path)
plot(result, "decoding.pdf","Decoding")



csv_path = "all.csv"  # 替换为你的 CSV 路径
result = parse_csv_to_result(csv_path)
plot(result, "all.pdf","Prefilling+Decoding")

