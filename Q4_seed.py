from sko.GA import GA
from sko.tools import set_run_mode
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import torch
from utils.Q4cal_mask_time import Q4_constraint_ueq, Q4_cal_mask_time
from utils.base import *
from utils.geo import *

# 设置随机种子
SEED = 92905  # 可以修改这个种子值 71267
MAX_ITER = 200  # 迭代次数

print(f"使用种子: {SEED}")

# 设置种子
np.random.seed(SEED)
torch.manual_seed(SEED)

# M1射中之前
max_time = math.sqrt(sum(MISSILES_INITIAL["M1"] ** 2)) / MISSILE_SPEED
print(f"max_time: {max_time}")


def func(x):
    return -Q4_cal_mask_time(x)


# 边界和约束
lb = [-140, -30, -140, -140, -140, 0] + [0] * 6
ub = [140, 30, 0, 140, 140, 140] + [max_time] * 6
ueq = (Q4_constraint_ueq,)

# 使用多进程加速
set_run_mode(func, "multiprocessing")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
print("开始优化...")
best_x, best_y = ga.run()

print("best_x:", best_x)
print("best_y:", best_y)

# 保存训练历史和绘图
Y_history = pd.DataFrame(ga.all_history_Y)
Y_history.to_csv(f"run/Q4_seed{SEED}_history.csv", index=False)
print("训练历史统计:")
print(Y_history.describe())

# 绘制损失曲线
fig, ax = plt.subplots(2, 1, figsize=(10, 8))
ax[0].plot(Y_history.index, Y_history.values, ".", color="red")
ax[0].set_title("所有个体的损失值")
ax[0].set_xlabel("迭代次数")
ax[0].set_ylabel("损失值")

Y_history.min(axis=1).cummin().plot(kind="line", ax=ax[1])
ax[1].set_title("最优损失值变化")
ax[1].set_xlabel("迭代次数")
ax[1].set_ylabel("最优损失值")

plt.tight_layout()

# 保存结果
plt.savefig(f"run/Q4_seed{SEED}_loss.png")
plt.show()

print(f"\n=== 最终结果 ===")
print(f"种子: {SEED}")
print(f"最优解: {best_x}")
print(f"最优值: {best_y}")
print(f"最终损失: {Y_history.min().min()}")
print(f"结果已保存到 run/Q4_seed{SEED}_history.csv 和 run/Q4_seed{SEED}_loss.png")
