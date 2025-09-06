import subprocess
import sys
import pandas as pd
import random
import json
import math
import multiprocessing

# 测试参数
N_SEEDS = 100  # 测试的种子数量
MAX_ITER = 10  # 每个种子的最大迭代次数

# 信号量，同时只允许一个subprocess执行
semaphore = multiprocessing.Semaphore(1)


def run_single_seed(seed):
    """使用subprocess运行单个种子的优化，使用信号量控制"""
    semaphore.acquire()  # 获取信号量，确保同时只有一个subprocess运行
    try:
        cmd = f"""
import sys
sys.path.append('.')
from sko.GA import GA
from sko.tools import set_run_mode
import numpy as np
import math
import pandas as pd
import torch
from utils.Q4cal_mask_time import Q4_constraint_ueq, Q4_cal_mask_time
from utils.base import *
from utils.geo import *
import json

# 设置种子
seed = {seed}
np.random.seed(seed)
torch.manual_seed(seed)

# 计算参数
max_time = math.sqrt(sum(MISSILES_INITIAL["M1"] ** 2)) / MISSILE_SPEED

def func(x):
    return -Q4_cal_mask_time(x)

# 边界和约束
lb = [-140, -30, -140, -140, -140, 0] + [0] * 6
ub = [140, 30, 0, 140, 140, 140] + [max_time] * 6
ueq = (Q4_constraint_ueq,)

# 使用multiprocessing提升速度
set_run_mode(func, "multiprocessing")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 初始化GA
ga = GA(
    func=func,
    n_dim=12,
    size_pop=94,
    max_iter={MAX_ITER},
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
Y_history.to_csv(f"run/Q4_seed{{seed}}_history.csv", index=False)

# 输出结果
result = {{
    "seed": seed,
    "best_x": best_x.tolist(),
    "best_y": float(best_y),
    "final_loss": float(Y_history.min().min())
}}

print(json.dumps(result))
"""

        result = subprocess.run(
            [sys.executable, "-c", cmd], capture_output=True, text=True
        )
        if result.returncode == 0:
            return json.loads(result.stdout.strip().split("\n")[-1])
        else:
            return None
    finally:
        semaphore.release()  # 释放信号量


# 主程序
if __name__ == "__main__":
    # 生成随机种子列表
    random_seeds = [random.randint(0, 100000) for _ in range(N_SEEDS)]
    print(f"使用的随机种子: {random_seeds}")

    # 存储所有结果
    all_results = []
    best_overall_x = None
    best_overall_y = float("inf")

    for i, seed in enumerate(random_seeds):
        print(f"\n=== 测试种子 {seed} (第{i+1}/{N_SEEDS}次) ===")

        result = run_single_seed(seed)
        if result:
            all_results.append(result)
            print(result)

            # 更新全局最优
            if result["best_y"] < best_overall_y:
                best_overall_y = result["best_y"]
                best_overall_x = result["best_x"]
                print(f"发现更优解! 当前全局最优: {best_overall_y}")

    # 保存所有结果
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("run/Q4_all_seeds_results.csv", index=False)

    print(f"\n=== 最终结果 ===")
    print(f"全局最优解: {best_overall_x}")
    print(f"全局最优值: {best_overall_y}")
