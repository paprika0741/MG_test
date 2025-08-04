import pandas as pd
import ast
import matplotlib.pyplot as plt
import math

# 读取 CSV 文件
df = pd.read_csv("result_loss_output.csv")

# 获取所有 step 数量
unique_steps = sorted(df["step"].unique())
num_steps = len(unique_steps)

# 自动计算子图排列行列数
cols = 2
rows = math.ceil(num_steps / cols)

fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5), 
                         squeeze=False)
# fig.suptitle("Accuracy Curve for Each Step", fontsize=18)

# 用于收集所有图例
all_handles = {}
for i, step in enumerate(unique_steps):
    ax = axes[i // cols][i % cols]
    group = df[df["step"] == step]
    
    for idx, row in group.iterrows():
        layer = row["layer"]
        acc_list = ast.literal_eval(row["acc"])
        line, = ax.plot(range(len(acc_list)), acc_list, label=f"Layer {layer}")

        # 只保留每个 layer 的一个句柄
        if f"Layer {layer}" not in all_handles:
            all_handles[f"Layer {layer}"] = line

    ax.set_title(f"Step {step}", fontsize=16)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_ylim(0, 1)
    ax.grid(True)

# 清除空子图（如果 step 数量不是偶数）
for j in range(i + 1, rows * cols):
    fig.delaxes(axes[j // cols][j % cols])

# 添加统一图例在顶部
fig.legend(
    handles=list(all_handles.values()),
    labels=list(all_handles.keys()),
    loc='upper center',
    bbox_to_anchor=(0.5, 0.98),  # 比默认 upper center 稍微往下
    ncol=8,
    fontsize=11,
    frameon=True,                # 打开图例边框
    framealpha=1.0,              # 设置不透明
    edgecolor='gray'            # 可选：设置边框颜色
)

plt.tight_layout(rect=[0, 0, 1, 0.85])  # 留更多空间给 legend 和 title
plt.savefig("all_steps_accuracy_plot.pdf")

