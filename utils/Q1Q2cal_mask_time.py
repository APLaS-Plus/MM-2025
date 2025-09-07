import sympy as sp
from sympy import Interval, oo, Rational
import numpy as np
from utils.base import *
from utils.geo import *


def Q2_cal_mask_time(input_data):
    # unzip input data
    f1_vx, f1_vy, drop_t, bomb_t = input_data

    # define FY1 and M1
    FY1_init_position = DRONES_INITIAL["FY1"]
    f1_V = np.array([f1_vx, f1_vy, 0])

    M1_init_position = MISSILES_INITIAL["M1"]
    M1_V = calculate_velocity_vector(M1_init_position, FAKE_TARGET, MISSILE_SPEED)

    # cal drop time
    drop_position = calculate_position_with_velocity(FY1_init_position, f1_V, drop_t)

    # cal bomb time
    bomb_position = calculate_parabolic_trajectory(drop_position, f1_V, bomb_t)
    M1_position = calculate_position_with_velocity(
        M1_init_position, M1_V, drop_t + bomb_t
    )

    # cal mask time
    t = sp.Symbol("t", real=True)
    t_domain = Interval(0, SMOKE_EFFECTIVE_TIME, left_open=True)
    Mx, My, Mz = (
        M1_position[0] + M1_V[0] * t,
        M1_position[1] + M1_V[1] * t,
        M1_position[2] + M1_V[2] * t,
    )

    # 烟幕球心在时刻t的位置（垂直下沉）
    Sx, Sy, Sz = (
        bomb_position[0],
        bomb_position[1],
        bomb_position[2] - SMOKE_SINK_SPEED * t,
    )

    # 位置向量
    Mxv = sp.Matrix([Mx, My, Mz])  # 导弹位置向量
    Sv = sp.Matrix([Sx, Sy, Sz])  # 烟幕球心位置向量

    # print(f"Mxv: {Mxv}, Sv: {Sv}")

    # ================== 线段与球体相交的数学公式 ==================
    # 线段参数方程：P(s) = T + s * (Mt - T) = T + s * D，其中：
    # T = true_target（真目标）
    # D = Mt - T（方向向量，从真目标指向导弹）
    # E = T - St（真目标到球心向量）
    #
    # 相交条件：|P(s) - St|² ≤ R²
    # 即：|T + s*D - St|² = |E + s*D|² ≤ R²
    # 展开：s²|D|² + 2s(E·D) + |E|² ≤ R²

    true_target = sp.Matrix(TRUE_TARGET_CENTER.copy())  # 真目标位置向量

    D_vec = Mxv - true_target  # 方向向量（从真目标指向导弹）
    E_vec = true_target - Sv  # 真目标到球心向量

    # 计算二次不等式系数（避免过度simplify提升速度）
    D_dot_D = D_vec.dot(D_vec)  # |D|²
    E_dot_D = E_vec.dot(D_vec)  # E·D
    E_dot_E = E_vec.dot(E_vec)  # |E|²

    R = SMOKE_EFFECTIVE_RADIUS  # 烟幕有效遮蔽半径

    # 二次不等式：s²|D|² + 2s(E·D) + |E|² - R² ≤ 0
    # 设 a = |D|², b = 2(E·D), c = |E|² - R²
    a_coeff = D_dot_D
    b_coeff = 2 * E_dot_D
    c_coeff = E_dot_E - R**2

    # ================== 线段到球心最小距离法（分段解析优化）==================
    # s* = clamp(-b/(2a), 0, 1) 的分段实现，避免符号Min/Max运算
    # 假设 a > 0（|D|²恒正），分三种情况：
    # 1. b ≥ 0: s* = 0, g_min = c
    # 2. -2a ≤ b < 0: s* = -b/(2a), g_min = c - b²/(4a)
    # 3. b < -2a: s* = 1, g_min = a + b + c

    # 预计算关键表达式，避免重复计算
    a_plus_b_plus_c = a_coeff + b_coeff + c_coeff
    c_minus_b_sq_over_4a = c_coeff - b_coeff**2 / (4 * a_coeff)

    # 三段遮蔽条件的时间区间求解
    # 段1: b ≥ 0, 遮蔽条件 c ≤ 0
    cond1_mask = sp.solveset(c_coeff <= 0, t, domain=t_domain)
    cond1_time = sp.solveset(b_coeff >= 0, t, domain=t_domain)
    interval1 = cond1_mask.intersect(cond1_time)

    # 段2: -2a ≤ b < 0, 遮蔽条件 c - b²/(4a) ≤ 0
    cond2_mask = sp.solveset(c_minus_b_sq_over_4a <= 0, t, domain=t_domain)
    cond2_time_lower = sp.solveset(b_coeff + 2 * a_coeff >= 0, t, domain=t_domain)
    cond2_time_upper = sp.solveset(b_coeff < 0, t, domain=t_domain)
    cond2_time = cond2_time_lower.intersect(cond2_time_upper)
    interval2 = cond2_mask.intersect(cond2_time)

    # 段3: b < -2a, 遮蔽条件 a + b + c ≤ 0
    cond3_mask = sp.solveset(a_plus_b_plus_c <= 0, t, domain=t_domain)
    cond3_time = sp.solveset(b_coeff + 2 * a_coeff < 0, t, domain=t_domain)
    interval3 = cond3_mask.intersect(cond3_time)

    # 合并所有有效区间
    intersection_intervals = interval1.union(interval2).union(interval3)

    # 返回有效遮蔽时长
    if intersection_intervals != sp.S.EmptySet and intersection_intervals.measure != 0:
        return intersection_intervals.measure
    else:
        return 0


def Q2_cal_mask_time_optimized(input_data, num_samples=2000):
    """
    使用采样法的优化版本计算遮蔽时间（速度更快）

    Args:
        input_data: (f1_vx, drop_t, bomb_t) 输入参数
        num_samples: 采样点数量，默认2000

    Returns:
        float: 遮蔽时间长度
    """
    # unzip input data
    f1_vx, f1_vy, drop_t, bomb_t = input_data

    # define FY1 and M1
    FY1_init_position = DRONES_INITIAL["FY1"]
    f1_V = np.array([f1_vx, f1_vy, 0])

    M1_init_position = MISSILES_INITIAL["M1"]
    M1_V = calculate_velocity_vector(M1_init_position, FAKE_TARGET, MISSILE_SPEED)

    # cal drop time
    drop_position = calculate_position_with_velocity(FY1_init_position, f1_V, drop_t)

    # cal bomb time
    bomb_position = calculate_parabolic_trajectory(drop_position, f1_V, bomb_t)
    M1_position = calculate_position_with_velocity(
        M1_init_position, M1_V, drop_t + bomb_t
    )

    # 采样法计算
    true_target = np.array(TRUE_TARGET_CENTER)
    R = SMOKE_EFFECTIVE_RADIUS

    # 时间采样
    t_start = 0.0
    t_end = SMOKE_EFFECTIVE_TIME
    time_samples = np.linspace(t_start, t_end, num_samples)

    intersection_flags = []  # 记录每个时间点是否相交

    for t in time_samples:
        # 导弹在时刻t的位置
        Mt = np.array(
            [
                M1_position[0] + M1_V[0] * t,
                M1_position[1] + M1_V[1] * t,
                M1_position[2] + M1_V[2] * t,
            ]
        )

        # 烟幕球心在时刻t的位置
        St = np.array(
            [
                bomb_position[0],
                bomb_position[1],
                bomb_position[2] - SMOKE_SINK_SPEED * t,
            ]
        )

        # 判断线段（真目标到导弹）是否与烟幕球相交
        is_intersecting = is_line_intersecting_sphere(true_target, Mt, St, R)
        intersection_flags.append(is_intersecting)

    # 统计相交时间区间
    intersection_intervals = []
    in_intersection = False
    start_time = None

    for i, flag in enumerate(intersection_flags):
        t = time_samples[i]

        if flag and not in_intersection:
            # 开始相交
            in_intersection = True
            start_time = t
        elif not flag and in_intersection:
            # 结束相交
            in_intersection = False
            intersection_intervals.append((start_time, t))

    # 处理结尾情况
    if in_intersection:
        intersection_intervals.append((start_time, t_end))

    # 计算总相交时长
    total_duration = sum(end - start for start, end in intersection_intervals)

    return total_duration


Q2_constraint_ueq = lambda x: -1 if 70.0**2 <= x[0]**2 + x[1]**2 <= 140.0**2 and x[3] > x[2] else 1
