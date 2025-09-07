"""
Q5计算模块
多无人机多导弹烟幕遮蔽时间计算

支持5架无人机对3枚导弹的组合遮蔽策略优化
每架无人机最多投放3枚烟幕干扰弹
"""

import sympy as sp
from sympy import Interval, oo, Union, EmptySet
import numpy as np
from utils.base import *
from utils.geo import *
from utils.Q4cal_mask_time import (
    simgle_f_cal_mask_time,
    simgle_f_cal_mask_time_optimized,
)


def single_drone_multi_smoke_mask_time(drone_assignment):
    """
    计算单个无人机投放多枚烟幕弹对指定导弹的遮蔽时间

    Args:
        drone_assignment: 字典格式
        {
            'drone_num': 1,
            'missile_target': 'M1',
            'flight_params': [vx, vy],  # 飞行参数
            'smoke_bombs': [
                {'drop_t': 1.5, 'bomb_t': 3.6},
                {'drop_t': 2.5, 'bomb_t': 4.6},
                {'drop_t': 3.5, 'bomb_t': 5.6}
            ]
        }

    Returns:
        sympy.Interval: 遮蔽时间区间
    """
    drone_num = drone_assignment["drone_num"]
    vx, vy = drone_assignment["flight_params"]
    smoke_bombs = drone_assignment["smoke_bombs"]

    # 计算每颗烟幕弹的遮蔽区间
    intervals = []
    for smoke in smoke_bombs:
        drop_t = smoke["drop_t"]
        bomb_t = smoke["bomb_t"]

        # 使用Q4的单个计算函数
        interval = simgle_f_cal_mask_time((drone_num, vx, vy, drop_t, bomb_t))
        if interval != EmptySet:
            intervals.append(interval)

    # 计算所有烟幕弹遮蔽区间的并集
    if not intervals:
        return EmptySet

    combined_interval = intervals[0]
    for interval in intervals[1:]:
        combined_interval = combined_interval.union(interval)

    return combined_interval


def single_drone_multi_smoke_mask_time_optimized(drone_assignment, num_samples=2000):
    """
    使用采样法的优化版本计算单个无人机投放多枚烟幕弹对指定导弹的遮蔽时间（速度更快）

    Args:
        drone_assignment: 字典格式（同上）
        num_samples: 采样点数量，默认2000

    Returns:
        list: 相交时间区间列表 [(start1, end1), (start2, end2), ...]
    """
    drone_num = drone_assignment["drone_num"]
    vx, vy = drone_assignment["flight_params"]
    smoke_bombs = drone_assignment["smoke_bombs"]

    # 计算每颗烟幕弹的遮蔽区间
    all_intervals = []
    for smoke in smoke_bombs:
        drop_t = smoke["drop_t"]
        bomb_t = smoke["bomb_t"]

        # 使用Q4的优化单个计算函数
        intervals = simgle_f_cal_mask_time_optimized(
            (drone_num, vx, vy, drop_t, bomb_t), num_samples
        )
        all_intervals.extend(intervals)

    return all_intervals


def merge_intervals_q5(intervals):
    """
    合并重叠的时间区间（Q5版本）

    Args:
        intervals: 时间区间列表 [(start1, end1), (start2, end2), ...]

    Returns:
        float: 合并后的总时间长度
    """
    if not intervals:
        return 0.0

    # 按起始时间排序
    sorted_intervals = sorted(intervals)
    merged = [sorted_intervals[0]]

    for current in sorted_intervals[1:]:
        last = merged[-1]

        # 检查是否重叠
        if current[0] <= last[1]:
            # 合并区间
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            # 不重叠，添加新区间
            merged.append(current)

    # 计算总时长
    total_duration = sum(end - start for start, end in merged)
    return total_duration


def calculate_multi_drone_mask_time(drone_assignments, target_missile):
    """
    计算多个无人机对指定导弹的总遮蔽时间

    Args:
        drone_assignments: 无人机分配列表
        target_missile: 目标导弹编号 ('M1', 'M2', 'M3')

    Returns:
        float: 总遮蔽时间
    """
    # 筛选出针对目标导弹的无人机
    relevant_assignments = [
        assignment
        for assignment in drone_assignments
        if assignment.get("missile_target") == target_missile
    ]

    if not relevant_assignments:
        return 0.0

    # 计算每个无人机的遮蔽区间
    intervals = []
    for assignment in relevant_assignments:
        interval = single_drone_multi_smoke_mask_time(assignment)
        if interval != EmptySet:
            intervals.append(interval)

    if not intervals:
        return 0.0

    # 计算所有区间的并集
    combined_interval = intervals[0]
    for interval in intervals[1:]:
        combined_interval = combined_interval.union(interval)

    return float(combined_interval.measure)


def calculate_multi_drone_mask_time_optimized(
    drone_assignments, target_missile, num_samples=2000
):
    """
    使用采样法的优化版本计算多个无人机对指定导弹的总遮蔽时间（速度更快）

    Args:
        drone_assignments: 无人机分配列表
        target_missile: 目标导弹编号 ('M1', 'M2', 'M3')
        num_samples: 采样点数量，默认2000

    Returns:
        float: 总遮蔽时间
    """
    # 筛选出针对目标导弹的无人机
    relevant_assignments = [
        assignment
        for assignment in drone_assignments
        if assignment.get("missile_target") == target_missile
    ]

    if not relevant_assignments:
        return 0.0

    # 计算每个无人机的遮蔽区间
    all_intervals = []
    for assignment in relevant_assignments:
        intervals = single_drone_multi_smoke_mask_time_optimized(
            assignment, num_samples
        )
        all_intervals.extend(intervals)

    if not all_intervals:
        return 0.0

    # 计算合并后的总时长
    return merge_intervals_q5(all_intervals)


def calculate_total_mask_effectiveness(drone_assignments):
    """
    计算所有无人机对所有导弹的总遮蔽效果

    Args:
        drone_assignments: 无人机分配列表

    Returns:
        dict: {'M1': mask_time, 'M2': mask_time, 'M3': mask_time, 'total': total_time}
    """
    results = {}
    total_time = 0.0

    for missile in ["M1", "M2", "M3"]:
        mask_time = calculate_multi_drone_mask_time(drone_assignments, missile)
        results[missile] = mask_time
        total_time += mask_time

    results["total"] = total_time
    return results


def calculate_total_mask_effectiveness_optimized(drone_assignments, num_samples=2000):
    """
    使用采样法的优化版本计算所有无人机对所有导弹的总遮蔽效果（速度更快）

    Args:
        drone_assignments: 无人机分配列表
        num_samples: 采样点数量，默认2000

    Returns:
        dict: {'M1': mask_time, 'M2': mask_time, 'M3': mask_time, 'total': total_time}
    """
    results = {}
    total_time = 0.0

    for missile in ["M1", "M2", "M3"]:
        mask_time = calculate_multi_drone_mask_time_optimized(
            drone_assignments, missile, num_samples
        )
        results[missile] = mask_time
        total_time += mask_time

    results["total"] = total_time
    return results


def optimize_single_drone_for_missile(
    drone_num, missile_target, max_smoke_bombs=3, excluded_intervals=None
):
    """
    为单个无人机优化对指定导弹的烟幕投放策略

    Args:
        drone_num: 无人机编号 (1-5)
        missile_target: 目标导弹 ('M1', 'M2', 'M3')
        max_smoke_bombs: 最大烟幕弹数量
        excluded_intervals: 其他无人机已占用的时间区间

    Returns:
        dict: 最优分配方案
    """
    # 根据导弹目标调整搜索边界
    max_time = get_missile_flight_time(missile_target)

    # 这里应该调用PSO优化算法，类似Q4的实现
    # 为简化，先返回一个示例结果
    best_assignment = {
        "drone_num": drone_num,
        "missile_target": missile_target,
        "flight_params": [100.0, 0.0],  # 示例飞行参数
        "smoke_bombs": [
            {"drop_t": 1.0, "bomb_t": 2.0},
            {"drop_t": 2.5, "bomb_t": 4.0},
            {"drop_t": 4.0, "bomb_t": 6.0},
        ][:max_smoke_bombs],
    }

    return best_assignment


def get_missile_flight_time(missile_name):
    """
    计算导弹飞行时间

    Args:
        missile_name: 导弹名称 ('M1', 'M2', 'M3')

    Returns:
        float: 飞行时间
    """
    missile_pos = MISSILES_INITIAL[missile_name]
    flight_time = np.sqrt(np.sum(missile_pos**2)) / MISSILE_SPEED
    return flight_time


def evaluate_assignment_effectiveness(drone_assignments):
    """
    评估无人机分配方案的整体效果

    Args:
        drone_assignments: 无人机分配列表

    Returns:
        float: 总体效果评分
    """
    mask_results = calculate_total_mask_effectiveness(drone_assignments)

    # 考虑均衡性的评分函数
    missile_times = [mask_results["M1"], mask_results["M2"], mask_results["M3"]]
    total_time = sum(missile_times)

    # 加入均衡性奖励：避免某个导弹完全没有防护
    min_time = min(missile_times)
    balance_bonus = min_time * 0.5  # 均衡性奖励

    return total_time + balance_bonus


# 约束检查函数
def check_drone_constraints(drone_assignment):
    """
    检查无人机分配是否满足约束条件

    Args:
        drone_assignment: 单个无人机分配方案

    Returns:
        bool: 是否满足约束
    """
    vx, vy = drone_assignment["flight_params"]
    smoke_bombs = drone_assignment["smoke_bombs"]

    # 速度约束：70 <= |v| <= 140
    v_magnitude = np.sqrt(vx**2 + vy**2)
    if not (70.0 <= v_magnitude <= 140.0):
        return False

    # 时间约束检查
    max_flight_time = get_missile_flight_time(drone_assignment["missile_target"])

    for i, smoke in enumerate(smoke_bombs):
        drop_t = smoke["drop_t"]
        bomb_t = smoke["bomb_t"]

        # 基本时间约束
        if drop_t < 0 or bomb_t < 0 or drop_t > bomb_t:
            return False

        if drop_t + bomb_t > max_flight_time:
            return False

        # 烟幕弹间隔约束：至少间隔1秒
        if i > 0:
            prev_drop_t = smoke_bombs[i - 1]["drop_t"]
            if drop_t - prev_drop_t < 1.0:
                return False

    return True


def get_drone_missile_priority_matrix():
    """
    获取无人机-导弹优先级矩阵
    基于几何位置分析各无人机对各导弹的适配性

    Returns:
        dict: {(drone_num, missile): priority_score}
    """
    priority_matrix = {}

    # 基于距离和角度分析适配性
    for drone_num in range(1, 6):  # FY1-FY5
        drone_pos = DRONES_INITIAL[f"FY{drone_num}"]

        for missile_name in ["M1", "M2", "M3"]:
            missile_pos = MISSILES_INITIAL[missile_name]

            # 计算无人机到导弹轨迹的适配性评分
            distance = np.linalg.norm(drone_pos - missile_pos)

            # 距离越近，优先级越高（反比例）
            distance_score = 1.0 / (1.0 + distance / 10000)

            # 考虑几何位置的适配性
            position_score = calculate_position_compatibility(drone_pos, missile_pos)

            # 综合评分
            priority_matrix[(drone_num, missile_name)] = distance_score * position_score

    return priority_matrix


def calculate_position_compatibility(drone_pos, missile_pos):
    """
    计算无人机位置与导弹轨迹的几何适配性

    Args:
        drone_pos: 无人机位置
        missile_pos: 导弹位置

    Returns:
        float: 适配性评分 (0-1)
    """
    # 考虑无人机相对于导弹-目标连线的位置
    target_pos = np.array(TRUE_TARGET_CENTER)

    # 导弹到目标的方向向量
    missile_to_target = target_pos - missile_pos
    missile_direction = missile_to_target / np.linalg.norm(missile_to_target)

    # 无人机到导弹轨迹的垂直距离
    drone_to_missile = drone_pos - missile_pos
    perpendicular_distance = np.linalg.norm(
        drone_to_missile
        - np.dot(drone_to_missile, missile_direction) * missile_direction
    )

    # 距离越小，适配性越高
    compatibility = 1.0 / (1.0 + perpendicular_distance / 5000)

    return compatibility
