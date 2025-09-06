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
    t_domain = Interval(0, oo, left_open=False)
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

    return proj_intervals | M_intervals


def Q3_cal_mask_time(input_data):
    f1_vx, drop_t1, bomb_t1, drop_t2, bomb_t2, drop_t3, bomb_t3 = input_data
    mask_time1 = simgle_cal_mask_time((f1_vx, drop_t1, bomb_t1))
    mask_time2 = simgle_cal_mask_time((f1_vx, drop_t2, bomb_t2))
    mask_time3 = simgle_cal_mask_time((f1_vx, drop_t3, bomb_t3))
    return (mask_time1 | mask_time2 | mask_time3).measure


vec_ueq = lambda x: 70.0 <= abs(x[0]) <= 140
time_ueq = lambda x: x[0] <= x[1] and x[0] + x[1] <= 66.99917080747261

Q3_constraint_ueq = lambda x: -1 if vec_ueq(x) and all(time_ueq(x[i*2+1:i*2+2+1]) for i in range(0, 3)) else 1


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
            if smoke_t < 0:
                continue  # 烟幕球还未爆炸

            # 导弹在时刻t的位置 (Mt)
            Mt = smoke_ball["M1_position"] + M1_V * smoke_t
            # 烟幕球心在时刻t的位置 (St) - 从爆炸位置开始下沉
            St = smoke_ball["bomb_position"] - np.array(
                [0, 0, SMOKE_SINK_SPEED * smoke_t]
            )

            # 计算几何参数
            Mt_pow2 = np.dot(Mt, Mt)  # |Mt|² - 导弹位置向量模长平方
            StMt = np.dot(St, Mt)  # St·Mt - 烟幕球心与导弹位置的点积
            St_pow2 = np.dot(St, St)  # |St|² - 烟幕球心位置向量模长平方

            if Mt_pow2 < 1e-12:  # 导弹在原点的特殊情况
                if np.sqrt(St_pow2) <= R:
                    return True
                continue

            # 线段上距离球心最近的点的参数
            u = StMt / Mt_pow2

            # 情况1：线段投影相交（最常见情况）
            # 要求：u ∈ [0,1] 且 线段到球心的最短距离 ≤ R
            if 0 <= u <= 1:
                # 线段到球心的最短距离平方 = |St|² - (St·Mt)²/|Mt|²
                dist_sq = St_pow2 - (StMt**2 / Mt_pow2)
                if dist_sq <= R**2:
                    return True

            # 情况2：u>1, 终点（导弹位置）在球内
            Mt_St_diff = Mt - St  # Mt - St 向量
            if np.dot(Mt_St_diff, Mt_St_diff) <= R**2:  # |Mt - St|² ≤ R²
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
