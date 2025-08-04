def dynamic_budget(B, L, K, E):
    # Step 1: 计算反比权重（前面层权重更高）
    weights = [1 / l for l in range(1, L + 1)]
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

total_result_dynamic = dict()
for B in range (26*6, 26*20+1):
    budgets = dynamic_budget(B,26,6,64)
    print(budgets)
    print(sum(list(budgets.values())))
   
 