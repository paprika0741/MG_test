import pandas as pd

# 修改为你的 CSV 文件路径
for l in range(1,7):
    csv_path = f"prefix_routing_L{l}_eval.csv"
    print("L = ", l)
    # 读取 CSV 文件
    df = pd.read_csv(csv_path)

    # 打印每列的平均值
    print("Average Evaluation Metrics:")
    print(f"Precision: {df['precision'].mean():.4f}")
    print(f"Recall:    {df['recall'].mean():.4f}")
    print(f"Jaccard:   {df['jaccard'].mean():.4f}")
