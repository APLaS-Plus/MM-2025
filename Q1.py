import sympy as sp
from sympy import Interval, oo, Rational, simplify, S
import numpy as np
from utils.base import *
from utils.geo import *

# get initial phisical parameters
FY1_init_position = DRONES_INITIAL["FY1"].copy()
print(f"FY1_init_position: {FY1_init_position}")
FY_target = FAKE_TARGET.copy()
FY_target[2] = FY1_init_position[2]
print(f"FY_target: {FY_target}")

FY1_V = calculate_velocity_vector(FY1_init_position, FY_target, Q1_FY1_SPEED)  # m/s
print(f"FY1_V: {FY1_V}")

M1_init_position = MISSILES_INITIAL["M1"].copy()
print(f"M1_init_position: {M1_init_position}")
print(f"FAKE_TARGET: {FAKE_TARGET}")

M1_V = calculate_velocity_vector(M1_init_position, FAKE_TARGET, MISSILE_SPEED)
print(f"M1_V: {M1_V}")

# 1.5s
FY1_position = calculate_position_with_velocity(
    FY1_init_position, FY1_V, Q1_LAUNCH_TIME
)
print(f"FY1_position: {FY1_position}")
# 3.6s
M1_position = calculate_position_with_velocity(
    M1_init_position, M1_V, Q1_LAUNCH_TIME + Q1_IGNITE_INTERVAL
)

print(f"M1_position: {M1_position}")

# drop the dingzhen bomb
bomb_position = calculate_parabolic_trajectory(FY1_position, FY1_V, Q1_IGNITE_INTERVAL)

print(f"bomb_position: {bomb_position}")

# ================== 几何求解：线段与球体相交判断 ==================
# 问题描述：判断从真目标(0,200,0)到导弹位置Mt的线段是否与烟幕球体相交
#
# 数学模型：
# 1. 线段参数方程：P(s) = T + s * (Mt - T)，其中 s ∈ [0,1], T为真目标位置
# 2. 球体：以St为球心，R为半径
# 3. 求解：线段P(s)与球体的相交时间区间

true_target = sp.Matrix(TRUE_TARGET_CENTER.copy())  # 真目标位置向量

t = sp.Symbol("t", real=True)
t_domain = Interval(0, 20, left_open=True)

# 导弹在时刻t的位置（直线飞行）
Mx, My, Mz = (
    M1_position[0] + M1_V[0] * t,
    M1_position[1] + M1_V[1] * t,
    M1_position[2] + M1_V[2] * t,
)

# 烟幕球心在时刻t的位置（垂直下沉）
Sx, Sy, Sz = bomb_position[0], bomb_position[1], bomb_position[2] - SMOKE_SINK_SPEED * t

# 位置向量
Mt_vec = sp.Matrix([Mx, My, Mz])  # 导弹位置向量
St_vec = sp.Matrix([Sx, Sy, Sz])  # 烟幕球心位置向量

print(f"True target: {true_target}")
print(f"Missile position vector: Mt(t) = {Mt_vec}")
print(f"Smoke center vector: St(t) = {St_vec}")

# ================== 线段与球体相交的数学公式 ==================
# 线段参数方程：P(s) = T + s * (Mt - T) = T + s * D，其中：
# T = true_target（真目标）
# D = Mt - T（方向向量，从真目标指向导弹）
# E = T - St（真目标到球心向量）
#
# 相交条件：|P(s) - St|² ≤ R²
# 即：|T + s*D - St|² = |E + s*D|² ≤ R²
# 展开：s²|D|² + 2s(E·D) + |E|² ≤ R²

D_vec = Mt_vec - true_target  # 方向向量（从真目标指向导弹）
E_vec = true_target - St_vec  # 真目标到球心向量

# 计算二次不等式系数（避免过度simplify提升速度）
D_dot_D = D_vec.dot(D_vec)  # |D|²
E_dot_D = E_vec.dot(D_vec)  # E·D
E_dot_E = E_vec.dot(E_vec)  # |E|²

print(f"\n线段与球体相交计算:")
print(f"|D|² = {D_dot_D}")
print(f"E·D = {E_dot_D}")
print(f"|E|² = {E_dot_E}")

R = SMOKE_EFFECTIVE_RADIUS  # 烟幕有效遮蔽半径

# 二次不等式：s²|D|² + 2s(E·D) + |E|² - R² ≤ 0
# 设 a = |D|², b = 2(E·D), c = |E|² - R²
a_coeff = D_dot_D
b_coeff = 2 * E_dot_D
c_coeff = E_dot_E - R**2

print(f"\n二次不等式系数:")
print(f"a = {a_coeff}")
print(f"b = {b_coeff}")
print(f"c = {c_coeff}")

# ================== 线段到球心最小距离法（分段解析优化）==================
# s* = clamp(-b/(2a), 0, 1) 的分段实现，避免符号Min/Max运算
# 假设 a > 0（|D|²恒正），分三种情况：
# 1. b ≥ 0: s* = 0, g_min = c
# 2. -2a ≤ b < 0: s* = -b/(2a), g_min = c - b²/(4a)
# 3. b < -2a: s* = 1, g_min = a + b + c

print(f"\n线段到球心最小距离法（分段优化）:")

# 预计算关键表达式，避免重复计算
a_plus_b_plus_c = a_coeff + b_coeff + c_coeff
c_minus_b_sq_over_4a = c_coeff - b_coeff**2 / (4 * a_coeff)

# 三段遮蔽条件的时间区间求解
print(f"分段求解遮蔽条件 g_min(t) ≤ 0:")

# 段1: b ≥ 0, 遮蔽条件 c ≤ 0
cond1_mask = sp.solveset(c_coeff <= 0, t, domain=t_domain)
cond1_time = sp.solveset(b_coeff >= 0, t, domain=t_domain)
interval1 = cond1_mask.intersect(cond1_time)
print(f"段1 (b≥0, c≤0): {interval1}")

# 段2: -2a ≤ b < 0, 遮蔽条件 c - b²/(4a) ≤ 0
cond2_mask = sp.solveset(c_minus_b_sq_over_4a <= 0, t, domain=t_domain)
cond2_time_lower = sp.solveset(b_coeff + 2 * a_coeff >= 0, t, domain=t_domain)
cond2_time_upper = sp.solveset(b_coeff < 0, t, domain=t_domain)
cond2_time = cond2_time_lower.intersect(cond2_time_upper)
interval2 = cond2_mask.intersect(cond2_time)
print(f"段2 (-2a≤b<0, c-b²/(4a)≤0): {interval2}")

# 段3: b < -2a, 遮蔽条件 a + b + c ≤ 0
cond3_mask = sp.solveset(a_plus_b_plus_c <= 0, t, domain=t_domain)
cond3_time = sp.solveset(b_coeff + 2 * a_coeff < 0, t, domain=t_domain)
interval3 = cond3_mask.intersect(cond3_time)
print(f"段3 (b<-2a, a+b+c≤0): {interval3}")

# 合并所有有效区间
intersection_intervals = interval1.union(interval2).union(interval3)

print(f"\n相交时间区间求解:")
print(f"相交条件（判别式≥0）: {intersection_intervals}")
print(f"相交时间区间: {intersection_intervals}")

# 计算有效遮蔽时长
if intersection_intervals != sp.S.EmptySet and intersection_intervals.measure != 0:
    mask_duration = intersection_intervals.measure
    print(f"\n最终结果:")
    print(f"有效遮蔽时间区间: {intersection_intervals}")
    print(f"有效遮蔽时长: {float(mask_duration):.6f} 秒")
else:
    print(f"\n最终结果:")
    print(f"无有效遮蔽时间")

# FY1_init_position: [17800     0  1800]
# FY_target: [   0    0 1800]
# FY1_V: [-120.    0.    0.]
# M1_init_position: [20000     0  2000]
# FAKE_TARGET: [0 0 0]
# M1_V: [-298.51115706    0.          -29.85111571]
# FY1_position: [17620.     0.  1800.]
# M1_position: [18477.59309898     0.          1847.7593099 ]
# bomb_position: [17188.        0.     1736.496]
# True target: Matrix([[0], [200], [0]])
# Missile position vector: Mt(t) = Matrix([[18477.5930989787 - 298.511157062997*t], [0], [1847.75930989787 - 29.8511157062997*t]])
# Smoke center vector: St(t) = Matrix([[17188.0000000000], [0.0], [1736.496 - 3*t]])

# 线段与球体相交计算:
# |D|² = 90000.0*t**2 - 11141850.7453451*t + 344875661.19874
# E·D = -89.553347118899*t**2 + 5188189.38854801*t - 320841496.835847
# |E|² = (3*t - 1736.496)**2 + 295467344.0

# 二次不等式系数:
# a = 90000.0*t**2 - 11141850.7453451*t + 344875661.19874
# b = -179.106694237798*t**2 + 10376378.777096*t - 641682993.671693
# c = (3*t - 1736.496)**2 + 295467244.0

# 参数s的解集（s ∈ [0,1]）: ConditionSet(s, s**2*(90000.0*t**2 - 11141850.7453451*t + 344875661.19874) + s*(-179.106694237798*t**2 + 10376378.777096*t - 641682993.671693) + (3*t - 1736.496)**2 + 295467244.0 <= 0, Interval(0, 1))

# 相交时间区间求解:
# 相交条件（判别式≥0）: Interval(2.93789075964668, 4.34808815870863)
# 相交时间区间: Interval(2.93789075964668, 4.34808815870863)

# 最终结果:
# 有效遮蔽时间区间: Interval(2.93789075964668, 4.34808815870863)
# 有效遮蔽时长: 1.410197 秒
