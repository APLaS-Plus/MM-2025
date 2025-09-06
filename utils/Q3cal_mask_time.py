import sympy as sp
from sympy import Interval, oo, Rational
import numpy as np
from utils.base import *
from utils.geo import *


def simgle_cal_mask_time(input_data):
    # unzip input data
    f1_vx, drop_t, bomb_t = input_data

    # define FY1 and M1
    FY1_init_position = DRONES_INITIAL["FY1"]
    f1_V = np.array([f1_vx, 0, 0])

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

    # 计算二次不等式系数
    D_dot_D = D_vec.dot(D_vec).simplify()  # |D|²
    E_dot_D = E_vec.dot(D_vec).simplify()  # E·D
    E_dot_E = E_vec.dot(E_vec).simplify()  # |E|²

    R = SMOKE_EFFECTIVE_RADIUS  # 烟幕有效遮蔽半径

    # 二次不等式：s²|D|² + 2s(E·D) + |E|² - R² ≤ 0
    # 设 a = |D|², b = 2(E·D), c = |E|² - R²
    a_coeff = D_dot_D
    b_coeff = 2 * E_dot_D
    c_coeff = E_dot_E - R**2

    # 求解二次不等式的判别式
    discriminant = b_coeff**2 - 4 * a_coeff * c_coeff

    # 相交条件：判别式 ≥ 0
    intersection_condition = discriminant >= 0

    # 求解相交时间区间
    intersection_intervals = sp.solveset(intersection_condition, t, domain=t_domain)

    return intersection_intervals


def Q3_cal_mask_time(input_data):
    f1_vx, drop_t1, bomb_t1, drop_t2, bomb_t2, drop_t3, bomb_t3 = input_data
    mask_time1 = simgle_cal_mask_time((f1_vx, drop_t1, bomb_t1))
    mask_time2 = simgle_cal_mask_time((f1_vx, drop_t2, bomb_t2))
    mask_time3 = simgle_cal_mask_time((f1_vx, drop_t3, bomb_t3))
    return (mask_time1 | mask_time2 | mask_time3).measure


vec_ueq = lambda x: 70.0 <= abs(x[0]) <= 140
time_ueq = lambda x: x[0] <= x[1] and x[0] + x[1] <= 66.99917080747261

Q3_constraint_ueq = lambda x: (
    -1
    if vec_ueq(x) and all(time_ueq(x[i * 2 + 1 : i * 2 + 2 + 1]) for i in range(0, 3))
    else 1
)


def Q3_cal_mask_time_optimized(input_data):
    """
    优化版本的Q3烟幕遮蔽时间计算函数
    处理3个烟幕球的情况

    Args:
        input_data: (f1_vx, drop_t1, bomb_t1, drop_t2, bomb_t2, drop_t3, bomb_t3)

    Returns:
        总遮蔽时间长度 (秒)
    """
    # if not Q3_constraint_ueq(input_data):
    #     return 0
    # unzip input data
    f1_vx, drop_t1, bomb_t1, drop_t2, bomb_t2, drop_t3, bomb_t3 = input_data

    # define FY1 and M1
    FY1_init_position = DRONES_INITIAL["FY1"]
    f1_V = np.array([f1_vx, 0, 0])

    M1_init_position = MISSILES_INITIAL["M1"]
    M1_V = calculate_velocity_vector(M1_init_position, FAKE_TARGET, MISSILE_SPEED)

    # 计算3个烟幕球的位置和导弹在各个时刻的位置
    smoke_balls = []
    for drop_t, bomb_t in [(drop_t1, bomb_t1), (drop_t2, bomb_t2), (drop_t3, bomb_t3)]:
        drop_position = calculate_position_with_velocity(
            FY1_init_position, f1_V, drop_t
        )
        bomb_position = calculate_parabolic_trajectory(drop_position, f1_V, bomb_t)
        M1_position = calculate_position_with_velocity(
            M1_init_position, M1_V, drop_t + bomb_t
        )
        smoke_balls.append(
            {
                "bomb_position": bomb_position,
                "M1_position": M1_position,
                "total_time": drop_t + bomb_t,
            }
        )

    R = SMOKE_EFFECTIVE_RADIUS

    def check_intersection_at_time(t):
        """
        检查在时刻t是否满足线段与任一烟幕球相交条件
        实现与Q3_cal_mask_time相同的几何逻辑
        """
        if t < 0:
            return False

        # 检查是否与任一烟幕球相交
        for smoke_ball in smoke_balls:
            # 调整时间：烟幕球从爆炸时刻开始存在
            smoke_t = t - smoke_ball["total_time"]
            if smoke_t < 0 or smoke_t > SMOKE_EFFECTIVE_TIME:
                continue  # 烟幕球还未爆炸或已消失（爆炸后20秒消失）

            # 导弹在时刻t的位置 (Mt)
            Mt = smoke_ball["M1_position"] + M1_V * smoke_t
            # 烟幕球心在时刻t的位置 (St) - 从爆炸位置开始下沉
            St = smoke_ball["bomb_position"] - np.array(
                [0, 0, SMOKE_SINK_SPEED * smoke_t]
            )

            # 真目标位置
            true_target = np.array(TRUE_TARGET_CENTER)

            # 线段参数方程：P(s) = T + s * (Mt - T) = T + s * D
            # 相交条件：|T + s*D - St|² ≤ R²
            # 即：s²|D|² + 2s(E·D) + |E|² ≤ R²

            D = Mt - true_target  # 方向向量（从真目标指向导弹）
            E = true_target - St  # 真目标到球心向量

            # 计算二次不等式系数
            D_dot_D = np.dot(D, D)  # |D|²
            E_dot_D = np.dot(E, D)  # E·D
            E_dot_E = np.dot(E, E)  # |E|²

            # 二次不等式：s²|D|² + 2s(E·D) + |E|² - R² ≤ 0
            # 设 a = |D|², b = 2(E·D), c = |E|² - R²
            a = D_dot_D
            b = 2 * E_dot_D
            c = E_dot_E - R**2

            if abs(a) < 1e-12:  # 线性情况：导弹与真目标重合
                if abs(b) < 1e-12:
                    # 常数情况：检查真目标是否在球内
                    if c <= 0:
                        return True
                else:
                    # 线性不等式：bs + c ≤ 0
                    s_critical = -c / b
                    if 0 <= s_critical <= 1 and c <= 0:
                        return True
                continue

            # 二次不等式情况
            discriminant = b**2 - 4 * a * c

            if discriminant < 0:
                # 判别式小于0，无实数解
                continue

            # 计算二次不等式的解区间
            sqrt_discriminant = np.sqrt(discriminant)
            s1 = (-b - sqrt_discriminant) / (2 * a)
            s2 = (-b + sqrt_discriminant) / (2 * a)

            # 确保s1 <= s2
            if s1 > s2:
                s1, s2 = s2, s1

            # 检查解区间是否与[0,1]有交集
            if a > 0:
                # 开口向上，不等式解为[s1, s2]
                if s1 <= 1 and s2 >= 0:
                    return True
            else:
                # 开口向下，不等式解为(-∞, s1] ∪ [s2, +∞)
                if s1 >= 0 or s2 <= 1:
                    return True

        return False

    def find_intersection_intervals():
        """
        数值方法寻找所有相交时间区间
        """
        max_time = SMOKE_EFFECTIVE_TIME + max(
            smoke_ball["total_time"] for smoke_ball in smoke_balls
        )
        dt = 0.01  # 初始时间步长

        # 粗扫描找到可能的相交区间
        time_points = np.arange(0, max_time, dt)
        intersections = [check_intersection_at_time(t) for t in time_points]

        # 找到相交区间的粗略边界
        intervals = []
        in_intersection = False
        start_idx = 0

        for i, is_intersect in enumerate(intersections):
            if is_intersect and not in_intersection:
                # 开始相交
                in_intersection = True
                start_idx = i
            elif not is_intersect and in_intersection:
                # 结束相交
                in_intersection = False
                intervals.append((start_idx, i - 1))

        # 处理最后一个区间
        if in_intersection:
            intervals.append((start_idx, len(intersections) - 1))

        # 精确化区间边界
        total_time = 0.0
        fine_dt = 0.0001  # 精确搜索步长

        for start_idx, end_idx in intervals:
            # 精确寻找区间起始点
            coarse_start = max(0, time_points[start_idx] - dt)
            coarse_end = min(max_time, time_points[end_idx] + dt)

            # 二分法精确定位起始边界
            left, right = coarse_start, coarse_start + 2 * dt
            while right - left > fine_dt:
                mid = (left + right) / 2
                if check_intersection_at_time(mid):
                    right = mid
                else:
                    left = mid
            precise_start = right

            # 二分法精确定位结束边界
            left, right = coarse_end - 2 * dt, coarse_end
            while right - left > fine_dt:
                mid = (left + right) / 2
                if check_intersection_at_time(mid):
                    left = mid
                else:
                    right = mid
            precise_end = left

            if precise_end > precise_start:
                total_time += precise_end - precise_start

        return total_time

    return find_intersection_intervals()
