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
    t_domain = Interval(0, oo, left_open=False)
    Mx, My, Mz = (
        M1_position[0] + M1_V[0] * t,
        M1_position[1] + M1_V[1] * t,
        M1_position[2] + M1_V[2] * t,
    )

# 烟幕球心在时刻t的位置（垂直下沉）
    Sx, Sy, Sz = bomb_position[0], bomb_position[1], bomb_position[2] - SMOKE_SINK_SPEED * t

    # 位置向量
    Mxv = sp.Matrix([Mx, My, Mz])  # 导弹位置向量
    Sv = sp.Matrix([Sx, Sy, Sz])  # 烟幕球心位置向量

    # print(f"Mxv: {Mxv}, Sv: {Sv}")


    # ================== 线段与球体相交的数学公式 ==================
    # 设线段为 P(s) = s * Mt，s ∈ [0,1]
    # 球心到线段的最短距离平方 = |St - proj(St, Mt)|²
    # 其中 proj(St, Mt) = (St·Mt / Mt·Mt) * Mt

    Mt_pow2 = (Mxv.dot(Mxv)).simplify()  # |Mt|² - 导弹位置向量模长平方
    StMt = (Sv.dot(Mxv)).simplify()  # St·Mt - 烟幕球心与导弹位置的点积
    St_pow2 = (Sv.dot(Sv)).simplify()  # |St|² - 烟幕球心位置向量模长平方
    R = SMOKE_EFFECTIVE_RADIUS  # 烟幕有效遮蔽半径

    # 离路径距离
    d = Sv.cross(Mxv).norm() / Mxv.norm()

    # 线段上距离球心最近的点的参数
    u = StMt / Mt_pow2  # 如果u∈[0,1]，则最近点在线段上；否则最近点是线段端点

    # 提取系数并转换为精确的有理数表示
    Mt_pow2_coeffs = sp.Poly(Mt_pow2, t).all_coeffs()
    StMt_coeffs = sp.Poly(StMt, t).all_coeffs()

    # 重新构建精确的表达式
    Mt_pow2_exact = sum(
        Rational(str(float(coeff))) * t ** (len(Mt_pow2_coeffs) - 1 - i)
        for i, coeff in enumerate(Mt_pow2_coeffs)
    )
    StMt_exact = sum(
        Rational(str(float(coeff))) * t ** (len(StMt_coeffs) - 1 - i)
        for i, coeff in enumerate(StMt_coeffs)
    )

    u = StMt_exact / Mt_pow2_exact  # 使用精确表达式

    # ================== 相交条件分析 ==================
    # 线段与球体相交需要满足以下任一条件：
    u_cond1 = u >= 0
    u_cond2 = u <= 1

    # 条件1：线段投影相交（最常见情况）
    # 要求：u ∈ [0,1] 且 线段到球心的最短距离 ≤ R
    proj_cond = St_pow2 - (StMt**2 / Mt_pow2) <= R**2

    # 条件2：u<0, 起点（原点）在球内，不合理于是不考虑
    # origin_cond = c <= R**2  # |St|² ≤ R²，即原点到球心距离 ≤ R

    # 条件3：u>1, 终点（导弹位置）在球内
    M_cond = (Sv - Mxv).dot(Sv - Mxv) <= R**2  # |St - Mt|² ≤ R²

    # ================== 求解时间区间 ==================

    # 求解几何相交条件
    u_intervals1 = sp.solveset(u_cond1, t, domain=t_domain)
    u_intervals2 = sp.solveset(u_cond2, t, domain=t_domain)

    proj_intervals = sp.solveset(
        proj_cond, t, domain=u_intervals1 & u_intervals2
    )  # 线段与球相交
    # origin_intervals = sp.solveset(origin_cond, t, domain=t_domain)  # 原点在球内
    M_intervals = sp.solveset(M_cond, t, domain=t_domain)  # 导弹在球内

    return (proj_intervals | M_intervals).measure


Q2_constraint_ueq = lambda x: 70**2 <= x[0] ** 2 + x[1] ** 2 <= 140**2 and x[3] > x[2]


def Q2_cal_mask_time_optimized(input_data):
    """
    优化版本的烟幕遮蔽时间计算函数
    使用纯数值计算替代符号计算，大幅提升性能

    Args:
        input_data: (f1_vx, f1_vy, drop_t, bomb_t)

    Returns:
        遮蔽时间长度 (秒)
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

    # 数值计算优化版本
    # 导弹轨迹: P(t) = M1_position + M1_V * t
    # 烟幕球心: B(t) = bomb_position + [0, 0, -SMOKE_SINK_SPEED * t]
    # 相交条件: |P(t) - B(t)| <= R

    # 计算相对位置和速度
    relative_pos = M1_position - bomb_position  # 在t=0时导弹相对烟幕球心的位置
    relative_vel = M1_V - np.array([0, 0, -SMOKE_SINK_SPEED])  # 导弹相对烟幕球心的速度

    R = SMOKE_EFFECTIVE_RADIUS

    # 相交条件转换为二次不等式: |relative_pos + relative_vel * t|² <= R²
    # 展开: (relative_pos + relative_vel * t) · (relative_pos + relative_vel * t) <= R²
    # 得到: ||relative_pos||² + 2*(relative_pos · relative_vel)*t + ||relative_vel||²*t² <= R²
    # 重排: ||relative_vel||²*t² + 2*(relative_pos · relative_vel)*t + (||relative_pos||² - R²) <= 0

    a = np.dot(relative_vel, relative_vel)  # ||relative_vel||²
    b = 2 * np.dot(relative_pos, relative_vel)  # 2*(relative_pos · relative_vel)
    c = np.dot(relative_pos, relative_pos) - R**2  # ||relative_pos||² - R²

    # 求解二次不等式 at² + bt + c <= 0
    if abs(a) < 1e-12:  # 相对速度为0的情况
        if c <= 0:  # 导弹始终在球内
            return float("inf")
        else:  # 导弹始终在球外
            return 0.0

    # 计算判别式
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:  # 无实根，不相交
        return 0.0

    # 求解二次方程的根
    sqrt_discriminant = np.sqrt(discriminant)
    t1 = (-b - sqrt_discriminant) / (2 * a)
    t2 = (-b + sqrt_discriminant) / (2 * a)

    # 确保t1 <= t2
    if t1 > t2:
        t1, t2 = t2, t1

    # 只考虑t >= 0的时间段
    t_start = max(0, t1)
    t_end = max(0, t2)

    if t_end <= t_start:
        return 0.0

    # 额外检查：确保在有效遮蔽时间内
    effective_time = SMOKE_EFFECTIVE_TIME
    t_end = min(t_end, effective_time)

    if t_end <= t_start:
        return 0.0

    return t_end - t_start
