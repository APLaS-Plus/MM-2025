"""
Q4变量

FY1~3 三台无人机x, y速度
三个烟雾弹投射时机
三个烟雾弹爆炸时机
"""

import sympy as sp
from sympy import Interval, oo, Rational
import numpy as np
from utils.base import *
from utils.geo import *


def simgle_f_cal_mask_time(input_data):
    # unzip input data
    f_num, f_vx, f_vy, drop_t, bomb_t = input_data

    # define FY1 and M1
    FY_init_position = DRONES_INITIAL["FY" + str(f_num)]
    f_V = np.array([f_vx, f_vy, 0])

    M1_init_position = MISSILES_INITIAL["M1"]
    M1_V = calculate_velocity_vector(M1_init_position, FAKE_TARGET, MISSILE_SPEED)

    # cal drop time
    drop_position = calculate_position_with_velocity(FY_init_position, f_V, drop_t)

    # cal bomb time
    bomb_position = calculate_parabolic_trajectory(drop_position, f_V, bomb_t)
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

    return intersection_intervals


def Q4_cal_mask_time(input_data):
    (
        f1_vx,
        f1_vy,
        f2_vx,
        f2_vy,
        f3_vx,
        f3_vy,
        drop_t1,
        bomb_t1,
        drop_t2,
        bomb_t2,
        drop_t3,
        bomb_t3,
    ) = input_data
    mask_time1 = simgle_f_cal_mask_time((1, f1_vx, f1_vy, drop_t1, bomb_t1))
    mask_time2 = simgle_f_cal_mask_time((2, f2_vx, f2_vy, drop_t2, bomb_t2))
    mask_time3 = simgle_f_cal_mask_time((3, f3_vx, f3_vy, drop_t3, bomb_t3))
    return (mask_time1 | mask_time2 | mask_time3).measure


vec_ueq = lambda x: 70.0**2 <= x[0] ** 2 + x[1] ** 2 <= 140**2
time_ueq = lambda x: x[0] <= x[1] and x[0] + x[1] <= 66.99917080747261

Q4_constraint_ueq = lambda x: (
    -1
    if vec_ueq(x[0:2])
    and vec_ueq(x[2:4])
    and vec_ueq(x[4:6])
    and all(time_ueq(x[i * 2 : i * 2 + 2]) for i in range(3, 6))
    else 1
)
