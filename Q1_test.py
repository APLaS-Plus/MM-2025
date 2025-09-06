import sympy as sp
from sympy import Interval, oo, Rational, simplify, S
import numpy as np
from utils.base import *
from utils.geo import *

# ================== 可调节测试参数 ==================
# 这些参数可以随时修改用于测试不同场景
# 会覆盖utils/base.py中的默认值
#
# 题目1 (固定参数):
#   - FY1速度: 120 m/s (朝假目标方向)
#   - 投放时间: 1.5 s
#   - 起爆间隔: 3.6 s
#
# 题目2 (优化参数):
#   - FY1速度: 70-140 m/s (可调整飞行方向)
#   - 投放时间: 需优化
#   - 起爆间隔: 需优化
# ====================================================

# Q2
# best_x is  [-70.40359495   0.           1.71457692] best_y is [-4.44587076676170]

# FY1无人机飞行速度 (m/s) - 题目1中为120，题目2中需要优化在[70,140]范围内
TEST_FY1_SPEED = -70.40359495

# 受领任务后投放烟幕弹的时间 (s) - 题目1中为1.5秒，题目2中需要优化
TEST_LAUNCH_TIME = 0

# 投放后到起爆的时间间隔 (s) - 题目1中为3.6秒，题目2中需要优化
TEST_IGNITE_INTERVAL = 1.71457692

print(f"=== 当前测试参数 ===")
print(f"FY1飞行速度: {TEST_FY1_SPEED} m/s")
print(f"投放时间: {TEST_LAUNCH_TIME} s")
print(f"起爆间隔: {TEST_IGNITE_INTERVAL} s")
print("=" * 50)

# get initial phisical parameters
FY1_init_position = DRONES_INITIAL["FY1"].copy()
print(f"FY1_init_position: {FY1_init_position}")
FY_target = FAKE_TARGET.copy()
FY_target[2] = FY1_init_position[2]
print(f"FY_target: {FY_target}")

FY1_V = np.array([TEST_FY1_SPEED, 0, 0])
print(f"FY1_V: {FY1_V}")

M1_init_position = MISSILES_INITIAL["M1"].copy()
print(f"M1_init_position: {M1_init_position}")
print(f"FAKE_TARGET: {FAKE_TARGET}")

M1_V = calculate_velocity_vector(M1_init_position, FAKE_TARGET, MISSILE_SPEED)
print(f"M1_V: {M1_V}")

# 投放时刻的FY1位置
FY1_position = calculate_position_with_velocity(
    FY1_init_position, FY1_V, TEST_LAUNCH_TIME
)
print(f"drop_position: {FY1_position}")
# 起爆时刻的导弹位置
M1_position = calculate_position_with_velocity(
    M1_init_position, M1_V, TEST_LAUNCH_TIME + TEST_IGNITE_INTERVAL
)

print(f"M1_position: {M1_position}")

# 投放烟幕弹后的抛物线轨迹终点（起爆点）
bomb_position = calculate_parabolic_trajectory(
    FY1_position, FY1_V, TEST_IGNITE_INTERVAL
)

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

# 计算二次不等式系数
D_dot_D = D_vec.dot(D_vec).simplify()  # |D|²
E_dot_D = E_vec.dot(D_vec).simplify()  # E·D
E_dot_E = E_vec.dot(E_vec).simplify()  # |E|²

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

# 求解二次不等式的判别式
discriminant = b_coeff**2 - 4 * a_coeff * c_coeff
print(f"判别式 = {discriminant.simplify()}")

# 当判别式 ≥ 0 且 a > 0 时，不等式有解
# 解为：s ∈ [(-b - √Δ)/(2a), (-b + √Δ)/(2a)] ∩ [0,1]

s = sp.Symbol("s", real=True)
quadratic_ineq = a_coeff * s**2 + b_coeff * s + c_coeff <= 0

# 在参数域[0,1]内求解二次不等式
s_domain = Interval(0, 1)
s_solutions = sp.solveset(quadratic_ineq, s, domain=s_domain)

print(f"\n参数s的解集（s ∈ [0,1]）: {s_solutions}")

# 对于每个时刻t，检查是否存在s使得线段与球相交
# 即：存在s ∈ [0,1]使得二次不等式成立
intersection_condition = discriminant >= 0

# 求解相交时间区间
intersection_intervals = sp.solveset(intersection_condition, t, domain=t_domain)

print(f"\n相交时间区间求解:")
print(f"相交条件（判别式≥0）: {intersection_condition}")
print(f"相交时间区间: {intersection_intervals}")

# 计算有效遮蔽时长
if intersection_intervals != sp.S.EmptySet and intersection_intervals.measure != 0:
    mask_duration = intersection_intervals.measure
    final_mask_interval = intersection_intervals
    final_mask_duration = mask_duration
    print(f"\n最终结果:")
    print(f"有效遮蔽时间区间: {intersection_intervals}")
    print(f"有效遮蔽时长: {float(mask_duration):.6f} 秒")
else:
    final_mask_interval = sp.S.EmptySet
    final_mask_duration = 0
    print(f"\n最终结果:")
    print(f"无有效遮蔽时间")

# ================== 测试结果总结 ==================
print("\n" + "=" * 60)
print("                    测试结果总结")
print("=" * 60)
print(f"FY1飞行速度: {TEST_FY1_SPEED} m/s")
print(f"投放时间: {TEST_LAUNCH_TIME} s")
print(f"起爆间隔: {TEST_IGNITE_INTERVAL} s")
if final_mask_duration > 0:
    print(f"烟幕有效遮蔽时长: {float(final_mask_duration):.6f} 秒")
else:
    print("烟幕有效遮蔽时长: 0.000000 秒")
print("=" * 60)

# result
# FY1_init_position: [17800     0  1800]
# FY_target: [   0    0 1800]
# FY1_V: [-120.    0.    0.]
# M1_init_position: [20000     0  2000]
# FAKE_TARGET: [0 0 0]
# M1_V: [-298.51115706    0.          -29.85111571]
# FY1_position: [17620.     0.  1800.]
# M1_position: [18477.59309898     0.          1847.7593099 ]
# bomb_position: [17188.        0.     1736.496]
# Mt_pow2: 90000.0*t**2 - 11141850.7453451*t + 344835661.19874, StMt: 89.553347118899*t**2 - 5188189.38854801*t + 320801496.835847, St_pow2: (3*t - 1736.496)**2 + 295427344.0
# u: (8955334711889901*t**2 - 518818938854800700000*t + 32080149683584660000000)/(1000*(8999999999999997*t**2 - 1114185074534506500*t + 34483566119874010000))

# 各条件的时间区间求解结果:
# u >= 0: Interval.Lopen(0, 20)
# u <= 1: Interval.Lopen(0, 297683067839852900000/8991044665288107099 - 5000*sqrt(2680239374023253994243020027412574)/8991044665288107099)
# u ∈ [0,1]: Interval.Lopen(0, 297683067839852900000/8991044665288107099 - 5000*sqrt(2680239374023253994243020027412574)/8991044665288107099)
# 情况1：线段与球相交: Interval(2.54870812587137, 297683067839852900000/8991044665288107099 - 5000*sqrt(2680239374023253994243020027412574)/8991044665288107099)
# 情况3：导弹在球内: Interval(4.28924753987335, 4.34808815870175)
# 最终相交时间区间: Interval(2.54870812587137, 4.34808815870175)
# 最终相交时间区间长度: 1.79938003283037
