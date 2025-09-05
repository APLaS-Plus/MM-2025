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

# FY1无人机飞行速度 (m/s) - 题目1中为120，题目2中需要优化在[70,140]范围内
TEST_FY1_SPEED = -120

# 受领任务后投放烟幕弹的时间 (s) - 题目1中为1.5秒，题目2中需要优化
TEST_LAUNCH_TIME = 1.5

# 投放后到起爆的时间间隔 (s) - 题目1中为3.6秒，题目2中需要优化
TEST_IGNITE_INTERVAL = 3.6

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
print(f"FY1_position: {FY1_position}")
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
# 问题描述：判断从原点O(0,0,0)到导弹位置Mt的线段是否与烟幕球体相交
#
# 数学模型：
# 1. 线段参数方程：P(s) = s * Mt，其中 s ∈ [0,1]
# 2. 球体：以St为球心，R为半径
# 3. 求解：线段P(s)与球体的相交时间区间

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
print(f"Mt_pow2: {Mt_pow2}, StMt: {StMt}, St_pow2: {St_pow2}")

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

print(f"u: {u.simplify()}")

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


# ================== 求最终相交时间区间 ==================
# 线段与球体相交的充要条件是以下任一条件成立：
# 1. (u∈[0,1] 且 判别式≥0) 或 2. 原点在球内 或 3. 导弹在球内

# 输出各个条件的求解结果
print("\n各条件的时间区间求解结果:")
print("u >= 0:", u_intervals1)
print("u <= 1:", simplify(u_intervals2))
print("u ∈ [0,1]:", simplify(u_intervals1 & u_intervals2))
print("情况1：线段与球相交:", simplify(proj_intervals))
# print("情况2：原点在球内:", origin_intervals)
print("情况3：导弹在球内:", simplify(M_intervals))
final_mask_interval = simplify(proj_intervals | M_intervals)
final_mask_duration = simplify(final_mask_interval.measure)

print("最终相交时间区间:", final_mask_interval)
print("最终相交时间区间长度:", final_mask_duration)

# ================== 测试结果总结 ==================
print("\n" + "=" * 60)
print("                    测试结果总结")
print("=" * 60)
print(f"FY1飞行速度: {TEST_FY1_SPEED} m/s")
print(f"投放时间: {TEST_LAUNCH_TIME} s")
print(f"起爆间隔: {TEST_IGNITE_INTERVAL} s")
print(f"烟幕有效遮蔽时长: {float(final_mask_duration):.6f} 秒")
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
