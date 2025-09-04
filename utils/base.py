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


# ================== 辅助函数 ==================
def get_missile_position(missile_name, time):
    """
    计算导弹在某时刻的位置

    Args:
        missile_name: 导弹名称 ('M1', 'M2', 'M3')
        time: 时间 (s)

    Returns:
        导弹位置坐标 np.array([x, y, z])
    """
    initial_pos = MISSILES_INITIAL[missile_name]
    # 导弹飞向假目标
    direction = FAKE_TARGET - initial_pos
    direction = direction / np.linalg.norm(direction)

    return initial_pos + direction * MISSILE_SPEED * time


def get_drone_position(drone_name, time, velocity_vector):
    """
    计算无人机在某时刻的位置

    Args:
        drone_name: 无人机名称 ('FY1', 'FY2', etc.)
        time: 时间 (s)
        velocity_vector: 速度向量 np.array([vx, vy, vz])，其中vz=0（等高度飞行）

    Returns:
        无人机位置坐标 np.array([x, y, z])
    """
    initial_pos = DRONES_INITIAL[drone_name]
    return initial_pos + velocity_vector * time


def get_smoke_center_position(launch_pos, ignite_time, time_after_ignite):
    """
    计算烟幕云团中心在起爆后某时刻的位置

    Args:
        launch_pos: 投放位置 np.array([x, y, z])
        ignite_time: 从投放到起爆的时间 (s)
        time_after_ignite: 起爆后的时间 (s)

    Returns:
        烟幕云团中心位置 np.array([x, y, z])
    """
    # 自由落体阶段
    fall_distance = 0.5 * 9.8 * ignite_time**2
    ignite_pos = launch_pos.copy()
    ignite_pos[2] -= fall_distance

    # 起爆后匀速下沉
    sink_distance = SMOKE_SINK_SPEED * time_after_ignite
    current_pos = ignite_pos.copy()
    current_pos[2] -= sink_distance

    return current_pos


def calculate_distance_3d(pos1, pos2):
    """
    计算两点之间的三维距离

    Args:
        pos1: 第一个点的坐标 np.array([x, y, z])
        pos2: 第二个点的坐标 np.array([x, y, z])

    Returns:
        距离值 (m)
    """
    return np.linalg.norm(pos1 - pos2)
