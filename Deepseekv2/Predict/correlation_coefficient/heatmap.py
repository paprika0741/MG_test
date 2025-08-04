import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# 假设 res 是你已经生成好的字典
with open('iteration_prefilling_200_1.pkl', 'rb') as f:
    res = pickle.load(f)

# === 设定直方图参数 ===
num_layers = len(res)
bins = np.arange(-1.0, 1.0 + 0.1, 0.1)  # [-1.0, -0.9, ..., 1.0]
bin_centers = 0.5 * (bins[:-1] + bins[1:])  # [-0.95, -0.85, ..., 0.95]
xtick_labels = np.round(bin_centers, 2)

# === 构建频率矩阵 ===
H = []
for l in range(1, num_layers + 1):
    hist, _ = np.histogram(res[l], bins=bins)
    hist = hist / hist.sum()  # 转为频率
    H.append(hist)
H = np.array(H)
# === 可视化 Heatmap ===
plt.figure(figsize=(12, 6))
sns.heatmap(H, xticklabels=np.round(bin_centers, 2), yticklabels=[f'L{l}' for l in range(1, num_layers+1)],
            cmap='YlGnBu',  )
plt.xlabel('Pearson Correlation Coefficient')
plt.ylabel('Layer')
# plt.title('Heatmap of Pearson Correlation Distributions Across Layers')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("iteration_prefilling_200_1_heatmap.pdf")
