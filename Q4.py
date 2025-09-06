from sko.GA import GA
from sko.tools import set_run_mode
import numpy as np
import math
import pandas as pd
import torch
import random
from utils.Q4cal_mask_time import Q4_constraint_ueq, Q4_cal_mask_time
from utils.base import *
from utils.geo import *

# 测试参数
N_SEEDS = 10  # 测试的种子数量
MAX_ITER = 30  # 每个种子的最大迭代次数

# M1射中之前
max_time = math.sqrt(sum(MISSILES_INITIAL["M1"] ** 2)) / MISSILE_SPEED
print(f"max_time: {max_time}")


def func(x):
    return -Q4_cal_mask_time(x)


# 边界和约束
lb = [-140, -140, -140, -140, -140, 0] + [0] * 6
ub = [140, 140, 0, 140, 140, 140] + [max_time] * 6
ueq = (Q4_constraint_ueq,)

set_run_mode(func, "multiprocessing")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 存储所有结果
all_results = []
best_overall_x = None
best_overall_y = float("inf")

# 生成随机种子列表
random_seeds = [random.randint(0, 100000) for _ in range(N_SEEDS)]
print(f"使用的随机种子: {random_seeds}")

for i, seed in enumerate(random_seeds):
    print(f"\n=== 测试种子 {seed} (第{i+1}/{N_SEEDS}次) ===")

    # 设置随机种子
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 初始化GA
    ga = GA(
        func=func,
        n_dim=12,
        size_pop=94,
        max_iter=MAX_ITER,
        prob_mut=1 / 12,
        lb=lb,
        ub=ub,
        constraint_ueq=ueq,
        precision=1e-3,
    ).to(device)

    # 运行优化
    best_x, best_y = ga.run()

    # 保存训练历史
    Y_history = pd.DataFrame(ga.all_history_Y)
    Y_history.to_csv(f"run/Q4_seed{seed}_history.csv", index=False)

    # 记录结果
    result = {
        "seed": seed,
        "best_x": best_x.tolist(),
        "best_y": best_y,
        "final_loss": Y_history.min().min(),
    }
    all_results.append(result)

    print(f"种子 {seed} - 最优值: {best_y}")

    # 更新全局最优
    if best_y < best_overall_y:
        best_overall_y = best_y
        best_overall_x = best_x.copy()
        print(f"发现更优解! 当前全局最优: {best_overall_y}")

# 保存所有结果
results_df = pd.DataFrame(all_results)
results_df.to_csv("run/Q4_all_seeds_results.csv", index=False)

print(f"\n=== 最终结果 ===")
print(f"全局最优解: {best_overall_x}")
print(f"全局最优值: {best_overall_y}")
print(f"结果已保存到 run/Q4_all_seeds_results.csv")
