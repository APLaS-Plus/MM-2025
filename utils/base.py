# 2025年高教社杯全国大学生数学建模竞赛 A题 基础数据
# 烟幕干扰弹的投放策略

import numpy as np

# ================== 烟幕弹相关参数 ==================
# 烟幕云团下沉速度 (m/s)
SMOKE_SINK_SPEED = 3

# 烟幕有效遮蔽半径 (m)
SMOKE_EFFECTIVE_RADIUS = 10

# 烟幕有效遮蔽时间 (s)
SMOKE_EFFECTIVE_TIME = 20

# 无人机投放两枚烟幕干扰弹的最小间隔时间 (s)
MIN_LAUNCH_INTERVAL = 1

# ================== 导弹相关参数 ==================
# 导弹飞行速度 (m/s)
MISSILE_SPEED = 300

# ================== 目标相关参数 ==================
# 假目标位置（原点）
FAKE_TARGET = np.array([0, 0, 0])

# 真目标参数
TRUE_TARGET_CENTER = np.array([0, 200, 0])  # 真目标下底面圆心
TRUE_TARGET_RADIUS = 7  # 真目标半径 (m)
TRUE_TARGET_HEIGHT = 10  # 真目标高度 (m)

# ================== 初始位置信息 ==================
# 导弹初始位置
MISSILES_INITIAL = {
    "M1": np.array([20000, 0, 2000]),
    "M2": np.array([19000, 600, 2100]),
    "M3": np.array([18000, -600, 1900]),
}

# 无人机初始位置
DRONES_INITIAL = {
    "FY1": np.array([17800, 0, 1800]),
    "FY2": np.array([12000, 1400, 1400]),
    "FY3": np.array([6000, -3000, 700]),
    "FY4": np.array([11000, 2000, 1800]),
    "FY5": np.array([13000, -2000, 1300]),
}

# ================== 无人机参数 ==================
# 无人机速度范围 (m/s)
DRONE_MIN_SPEED = 70
DRONE_MAX_SPEED = 140

# 每架无人机最多投放烟幕弹数量
MAX_SMOKE_PER_DRONE = 3

# ================== 问题1特定参数 ==================
# FY1的飞行速度 (m/s)
Q1_FY1_SPEED = 120

# 受领任务后投放时间 (s)
Q1_LAUNCH_TIME = 1.5

# 投放后起爆时间间隔 (s)
Q1_IGNITE_INTERVAL = 3.6
