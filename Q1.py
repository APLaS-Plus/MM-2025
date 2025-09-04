import sympy as sp
import numpy as np
from utils.base import *
from utils.geo import *

# get initial phisical parameters
FY1_init_position = DRONES_INITIAL["FY1"]

FY1_V = calculate_velocity_vector(FY1_init_position, FAKE_TARGET, Q1_FY1_SPEED)  # m/s

M1_init_position = MISSILES_INITIAL["M1"]
M1_V = calculate_velocity_vector(M1_init_position, FAKE_TARGET, MISSILE_SPEED)

# 1.5s
FY1_position = calculate_position_with_velocity(
    FY1_init_position, FY1_V, Q1_LAUNCH_TIME
)

# 3.6s
M1_position = calculate_parabolic_trajectory(
    M1_init_position, M1_V, Q1_LAUNCH_TIME + Q1_IGNITE_INTERVAL
)

print(M1_position)

# drop the dingzhen bomb
bomb_position = calculate_parabolic_trajectory(FY1_position, FY1_V, Q1_IGNITE_INTERVAL)

print(bomb_position)

# ================== 几何求解：线段与球体相交判断 ==================
# 问题描述：判断从原点O(0,0,0)到导弹位置Mt的线段是否与烟幕球体相交
#
# 数学模型：
# 1. 线段参数方程：P(s) = s * Mt，其中 s ∈ [0,1]
# 2. 球体：以St为球心，R为半径
# 3. 求解：线段P(s)与球体的相交时间区间

t = sp.Symbol("t", real=True)

# 导弹在时刻t的位置（直线飞行）
Mx, My, Mz = (
    M1_position[0] + M1_V[0] * t,
    M1_position[1] + M1_V[1] * t,
    M1_position[2] + M1_V[2] * t,
)

# 烟幕球心在时刻t的位置（垂直下沉）
Bx, By, Bz = bomb_position[0], bomb_position[1], bomb_position[2] + SMOKE_SINK_SPEED * t

# 位置向量
Mxv = sp.Matrix([Mx, My, Mz])  # 导弹位置向量
Bv = sp.Matrix([Bx, By, Bz])  # 烟幕球心位置向量

# ================== 线段与球体相交的数学公式 ==================
# 设线段为 P(s) = s * Mt，s ∈ [0,1]
# 球心到线段的最短距离平方 = |St - proj(St, Mt)|²
# 其中 proj(St, Mt) = (St·Mt / Mt·Mt) * Mt

a = (Mxv.dot(Mxv)).simplify()  # |Mt|² - 导弹位置向量模长平方
b = (Bv.dot(Mxv)).simplify()  # St·Mt - 烟幕球心与导弹位置的点积
c = (Bv.dot(Bv)).simplify()  # |St|² - 烟幕球心位置向量模长平方
R = SMOKE_EFFECTIVE_RADIUS  # 烟幕有效遮蔽半径

# 线段上距离球心最近的点的参数
u = b / a  # 如果u∈[0,1]，则最近点在线段上；否则最近点是线段端点

# ================== 相交条件分析 ==================
# 线段与球体相交需要满足以下任一条件：

# 条件1：线段投影相交（最常见情况）
# 要求：u ∈ [0,1] 且 线段到球心的最短距离 ≤ R
u_cond1 = u >= 0  # 投影点不在起点之前
u_cond2 = u <= 1  # 投影点不在终点之后
discriminant_cond = b**2 - (c - R**2) * a >= 0  # 线段与球体确实相交

# 条件2：起点（原点）在球内
origin_cond = c <= R**2  # |St|² ≤ R²，即原点到球心距离 ≤ R

# 条件3：终点（导弹位置）在球内
M_cond = (Bv - Mxv).dot(Bv - Mxv) <= R**2  # |St - Mt|² ≤ R²

# ================== 求解时间区间 ==================
# 分别求解每个几何条件对应的时间不等式

print("开始求解线段与球体相交的时间区间...")

# 求解投影参数的有效范围
u_intervals1 = sp.solve_univariate_inequality(u_cond1, t)  # u ≥ 0的时间区间
u_intervals2 = sp.solve_univariate_inequality(u_cond2, t)  # u ≤ 1的时间区间

# 求解几何相交条件
discriminant_intervals = sp.solve_univariate_inequality(
    discriminant_cond, t
)  # 线段与球相交
origin_intervals = sp.solve_univariate_inequality(origin_cond, t)  # 原点在球内
M_intervals = sp.solve_univariate_inequality(M_cond, t)  # 导弹在球内

# 输出各个条件的求解结果
print("\n各条件的时间区间求解结果:")
print("u ≥ 0 (投影点在起点后):", u_intervals1)
print("u ≤ 1 (投影点在终点前):", u_intervals2)
print("判别式 ≥ 0 (线段与球相交):", discriminant_intervals)
print("原点在球内:", origin_intervals)
print("导弹在球内:", M_intervals)

# ================== 求最终相交时间区间 ==================
# 线段与球体相交的充要条件是以下任一条件成立：
# 1. (u∈[0,1] 且 判别式≥0) 或 2. 原点在球内 或 3. 导弹在球内
# 这里先输出各个分量，实际应用时需要求这些区间的并集


# 根据结果直接求解
# 各条件的时间区间求解结果:
# u ≥ 0 (投影点在起点后): (-57866.2240709373 <= t) & (t <= 61.8513828513126)
# u ≤ 1 (投影点在终点前): (61.8482697471805 <= t) | (t <= 4.27808855382667)
# 判别式 ≥ 0 (线段与球相交): ((-18.9428563377376 <= t) & (t <= -14.1553549656438)) | ((92.2193430345465 <= t) & (t <= 94.1320036948343))
# 原点在球内: False
# 导弹在球内: False

# 负数非法
t1_bg = -18.9428563377376
t1_ed = -14.1553549656438

t2_bg = 92.2193430345465
t2_ed = 94.1320036948343

result = t2_ed - t2_bg
print(result)