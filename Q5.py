"""
Q5
贪心+启发式搜索

5架无人机对3枚导弹的烟幕投放策略优化
每架无人机最多投放3枚烟幕干扰弹
"""

import sympy as sp
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import copy
import subprocess
import sys
import json
from utils.base import *
from utils.geo import *
from utils.Q5cal_mask_time import *

# 初始化数据
m_i = copy.deepcopy(MISSILES_INITIAL)
f_i = copy.deepcopy(DRONES_INITIAL)

# 计算各导弹的飞行时间
missile_flight_times = {}
for missile_name in ["M1", "M2", "M3"]:
    pos = MISSILES_INITIAL[missile_name]
    flight_time = np.sqrt(np.sum(pos**2)) / MISSILE_SPEED
    missile_flight_times[missile_name] = flight_time

print(f"导弹飞行时间: {missile_flight_times}")

"""
策略：贪心+PSO

1. 计算无人机-导弹适配性矩阵
2. 贪心选择最优无人机-导弹配对
3. 对每个配对使用PSO优化烟幕投放参数
4. 考虑已有遮蔽区间，最大化总遮蔽效果
"""


def get_drone_missile_search_bounds(drone_num, missile_target):
    """
    根据无人机编号和目标导弹获取合适的搜索边界
    基于几何位置分析，设置智能搜索边界提高PSO效率
    
    Args:
        drone_num: 无人机编号 (1-5)
        missile_target: 目标导弹 ('M1', 'M2', 'M3')
        
    Returns:
        (custom_lb, custom_ub): 自定义的下界和上界 [vx_min, vy_min, ...]
    """
    # 计算目标导弹的最大飞行时间
    missile_pos = MISSILES_INITIAL[missile_target]
    max_time = math.sqrt(sum(missile_pos ** 2)) / MISSILE_SPEED
    
    # 获取无人机位置
    drone_pos = DRONES_INITIAL[f"FY{drone_num}"]
    
    # 基本原则：
    # 1. 导弹都飞向假目标(0,0,0)，真目标在(0,200,0)
    # 2. 无人机需要在导弹路径和真目标之间制造拦截
    # 3. 根据无人机当前位置和导弹路径设置优先移动方向
    
    if missile_target == "M1":  # M1: (20000, 0, 2000) -> (0, 0, 0)
        if drone_num == 1:  # FY1(17800, 0, 1800) - 最接近M1路径
            custom_lb = [-140, -20, 0, 0] + [0, 0] * 2  # 允许小幅y方向移动
            custom_ub = [50, 20, max_time, max_time] + [max_time, max_time] * 2
        elif drone_num == 2:  # FY2(12000, 1400, 1400) - 中距离，y正方向
            custom_lb = [-50, -30, 0, 0] + [0, 0] * 2  # 主要向左移动，可向下
            custom_ub = [30, 10, max_time, max_time] + [max_time, max_time] * 2
        elif drone_num == 3:  # FY3(6000, -3000, 700) - 远距离，y负方向
            custom_lb = [-30, 0, 0, 0] + [0, 0] * 2  # 需要向左上移动
            custom_ub = [30, 140, max_time, max_time] + [max_time, max_time] * 2
        elif drone_num == 4:  # FY4(11000, 2000, 1800) - 中距离，y正方向  
            custom_lb = [-50, -40, 0, 0] + [0, 0] * 2  # 向左下移动
            custom_ub = [30, 20, max_time, max_time] + [max_time, max_time] * 2
        else:  # FY5(13000, -2000, 1300) - 中距离，y负方向
            custom_lb = [-50, 0, 0, 0] + [0, 0] * 2  # 向左上移动
            custom_ub = [30, 140, max_time, max_time] + [max_time, max_time] * 2
            
    elif missile_target == "M2":  # M2: (19000, 600, 2100) -> (0, 0, 0)
        if drone_num == 1:  # FY1(17800, 0, 1800)
            custom_lb = [-140, 0, 0, 0] + [0, 0] * 2  # 向左上移动接近M2路径
            custom_ub = [50, 50, max_time, max_time] + [max_time, max_time] * 2
        elif drone_num == 2:  # FY2(12000, 1400, 1400) - 已在合适的y位置
            custom_lb = [-50, -30, 0, 0] + [0, 0] * 2  # 主要向左移动
            custom_ub = [30, 30, max_time, max_time] + [max_time, max_time] * 2
        elif drone_num == 3:  # FY3(6000, -3000, 700) - 需要大幅向上移动
            custom_lb = [-30, 50, 0, 0] + [0, 0] * 2  # 强制向左上移动
            custom_ub = [30, 140, max_time, max_time] + [max_time, max_time] * 2
        elif drone_num == 4:  # FY4(11000, 2000, 1800) - 已在良好位置
            custom_lb = [-50, -20, 0, 0] + [0, 0] * 2  # 向左移动，轻微调整y
            custom_ub = [30, 30, max_time, max_time] + [max_time, max_time] * 2
        else:  # FY5(13000, -2000, 1300) - 需要向上移动
            custom_lb = [-50, 30, 0, 0] + [0, 0] * 2  # 向左上移动
            custom_ub = [30, 140, max_time, max_time] + [max_time, max_time] * 2
            
    else:  # M3: (18000, -600, 1900) -> (0, 0, 0)
        if drone_num == 1:  # FY1(17800, 0, 1800)
            custom_lb = [-140, -50, 0, 0] + [0, 0] * 2  # 向左下移动接近M3路径
            custom_ub = [50, 0, max_time, max_time] + [max_time, max_time] * 2
        elif drone_num == 2:  # FY2(12000, 1400, 1400) - 需要向下移动
            custom_lb = [-50, -140, 0, 0] + [0, 0] * 2  # 向左下移动
            custom_ub = [30, -10, max_time, max_time] + [max_time, max_time] * 2
        elif drone_num == 3:  # FY3(6000, -3000, 700) - 已在合适的y负方向
            custom_lb = [-30, -10, 0, 0] + [0, 0] * 2  # 向左移动，保持y负方向
            custom_ub = [30, 50, max_time, max_time] + [max_time, max_time] * 2
        elif drone_num == 4:  # FY4(11000, 2000, 1800) - 需要大幅向下移动
            custom_lb = [-50, -140, 0, 0] + [0, 0] * 2  # 强制向左下移动
            custom_ub = [30, -20, max_time, max_time] + [max_time, max_time] * 2
        else:  # FY5(13000, -2000, 1300) - 已在良好位置
            custom_lb = [-50, -20, 0, 0] + [0, 0] * 2  # 向左移动，轻微调整y
            custom_ub = [30, 30, max_time, max_time] + [max_time, max_time] * 2
            
    return custom_lb, custom_ub


def optimize_drone_for_missile_pso(
    drone_num, missile_target, excluded_intervals_data=None, max_smoke_bombs=3
):
    """
    使用PSO为单个无人机优化对指定导弹的多烟幕弹投放策略

    Args:
        drone_num: 无人机编号 (1-5)
        missile_target: 目标导弹 ('M1', 'M2', 'M3')
        excluded_intervals_data: 其他无人机已占用的时间区间
        max_smoke_bombs: 最大烟幕弹数量

    Returns:
        (best_assignment, best_coverage_time)
    """
    max_retries = 5

    for retry in range(max_retries):
        cmd = f"""
import sys
sys.path.append('.')
from sko.PSO import PSO
from sko.tools import set_run_mode
import numpy as np
import math
import json
import sympy as sp
from utils.Q5cal_mask_time import single_drone_multi_smoke_mask_time
from utils.base import *
from utils.geo import *

drone_num = {drone_num}
missile_target = "{missile_target}"
max_smoke_bombs = {max_smoke_bombs}
excluded_intervals_data = {repr(excluded_intervals_data)}

# 计算目标导弹的飞行时间
missile_pos = MISSILES_INITIAL[missile_target]
max_time = math.sqrt(sum(missile_pos ** 2)) / MISSILE_SPEED

# 解析已有区间数据
excluded_intervals = None
if excluded_intervals_data:
    excluded_intervals = []
    for interval_str in excluded_intervals_data:
        exec("from sympy import *")
        interval_obj = eval(interval_str)
        excluded_intervals.append(interval_obj)

def func(x):
    # x包含: [vx, vy, drop_t1, bomb_t1, drop_t2, bomb_t2, drop_t3, bomb_t3]
    n_params = len(x)
    vx, vy = x[0], x[1]
    
    # 构建烟幕弹数据
    smoke_bombs = []
    for i in range(max_smoke_bombs):
        if 2 + i*2 + 1 < n_params:  # 确保有足够的参数
            drop_t = x[2 + i*2]
            bomb_t = x[2 + i*2 + 1]
            smoke_bombs.append({{'drop_t': drop_t, 'bomb_t': bomb_t}})
    
    # 构建无人机分配
    drone_assignment = {{
        'drone_num': drone_num,
        'missile_target': missile_target,
        'flight_params': [vx, vy],
        'smoke_bombs': smoke_bombs
    }}
    
    try:
        # 计算当前无人机的遮蔽区间
        interval = single_drone_multi_smoke_mask_time(drone_assignment)
        
        if excluded_intervals is None or len(excluded_intervals) == 0:
            coverage_time = float(interval.measure) if interval.measure.is_real else 0.0
        else:
            # 计算与已有区间的并集增量
            combined_interval = interval
            for exist_interval in excluded_intervals:
                combined_interval = combined_interval.union(exist_interval)
            
            original_time = sum(float(intv.measure) for intv in excluded_intervals if intv.measure.is_real)
            total_time = float(combined_interval.measure) if combined_interval.measure.is_real else 0.0
            coverage_time = max(0, total_time - original_time)
        
        return -coverage_time  # PSO求最小值
        
    except Exception as e:
        return 0  # 发生错误时返回0

def constraint(x):
    n_params = len(x)
    vx, vy = x[0], x[1]
    
    # 速度约束
    v_magnitude_sq = vx**2 + vy**2
    if not (70.0**2 <= v_magnitude_sq <= 140.0**2):
        return 1
    
    # 时间约束
    for i in range(max_smoke_bombs):
        if 2 + i*2 + 1 < n_params:
            drop_t = x[2 + i*2]
            bomb_t = x[2 + i*2 + 1]
            
            if drop_t < 0 or bomb_t < 0 or drop_t > bomb_t:
                return 1
            if drop_t + bomb_t > max_time:
                return 1
            
            # 烟幕弹间隔约束
            if i > 0 and 2 + (i-1)*2 < n_params:
                prev_drop_t = x[2 + (i-1)*2]
                if drop_t - prev_drop_t < 1.0:
                    return 1
    
    return -1

# 智能边界设置函数（在子进程内定义）
def get_bounds(drone_num, missile_target):
    missile_pos = MISSILES_INITIAL[missile_target]
    max_t = math.sqrt(sum(missile_pos ** 2)) / MISSILE_SPEED
    
    if missile_target == "M1":
        if drone_num == 1:
            return [-140, -20, 0, 0] + [0, 0] * 2, [50, 20, max_t, max_t] + [max_t, max_t] * 2
        elif drone_num == 2:
            return [-50, -30, 0, 0] + [0, 0] * 2, [30, 10, max_t, max_t] + [max_t, max_t] * 2
        elif drone_num == 3:
            return [-30, 0, 0, 0] + [0, 0] * 2, [30, 140, max_t, max_t] + [max_t, max_t] * 2
        elif drone_num == 4:
            return [-50, -40, 0, 0] + [0, 0] * 2, [30, 20, max_t, max_t] + [max_t, max_t] * 2
        else:
            return [-50, 0, 0, 0] + [0, 0] * 2, [30, 140, max_t, max_t] + [max_t, max_t] * 2
    elif missile_target == "M2":
        if drone_num == 1:
            return [-140, 0, 0, 0] + [0, 0] * 2, [50, 50, max_t, max_t] + [max_t, max_t] * 2
        elif drone_num == 2:
            return [-50, -30, 0, 0] + [0, 0] * 2, [30, 30, max_t, max_t] + [max_t, max_t] * 2
        elif drone_num == 3:
            return [-30, 50, 0, 0] + [0, 0] * 2, [30, 140, max_t, max_t] + [max_t, max_t] * 2
        elif drone_num == 4:
            return [-50, -20, 0, 0] + [0, 0] * 2, [30, 30, max_t, max_t] + [max_t, max_t] * 2
        else:
            return [-50, 30, 0, 0] + [0, 0] * 2, [30, 140, max_t, max_t] + [max_t, max_t] * 2
    else:  # M3
        if drone_num == 1:
            return [-140, -50, 0, 0] + [0, 0] * 2, [50, 0, max_t, max_t] + [max_t, max_t] * 2
        elif drone_num == 2:
            return [-50, -140, 0, 0] + [0, 0] * 2, [30, -10, max_t, max_t] + [max_t, max_t] * 2
        elif drone_num == 3:
            return [-30, -10, 0, 0] + [0, 0] * 2, [30, 50, max_t, max_t] + [max_t, max_t] * 2
        elif drone_num == 4:
            return [-50, -140, 0, 0] + [0, 0] * 2, [30, -20, max_t, max_t] + [max_t, max_t] * 2
        else:
            return [-50, -20, 0, 0] + [0, 0] * 2, [30, 30, max_t, max_t] + [max_t, max_t] * 2

# 使用智能边界
custom_lb, custom_ub = get_bounds(drone_num, missile_target)
n_dim = len(custom_lb)
lb = custom_lb
ub = custom_ub

# PSO优化
w = 0.9
c = (1 - w) / 2
set_run_mode(func, "multiprocessing")
pso = PSO(func=func, n_dim=n_dim, pop=64, max_iter=150, lb=lb, ub=ub, 
          constraint_ueq=(constraint,), c1=c, c2=c, w=w)
pso.run()

best_params = pso.gbest_x
best_coverage_time = -pso.gbest_y.item()

# 构建最优分配结果
vx, vy = best_params[0], best_params[1]
smoke_bombs = []
for i in range(max_smoke_bombs):
    if 2 + i*2 + 1 < len(best_params):
        drop_t = best_params[2 + i*2]
        bomb_t = best_params[2 + i*2 + 1]
        smoke_bombs.append({{'drop_t': float(drop_t), 'bomb_t': float(bomb_t)}})

best_assignment = {{
    'drone_num': drone_num,
    'missile_target': missile_target,
    'flight_params': [float(vx), float(vy)],
    'smoke_bombs': smoke_bombs
}}

result = {{
    "best_assignment": best_assignment,
    "best_coverage_time": float(best_coverage_time)
}}

print(json.dumps(result))
"""

        try:
            result = subprocess.run(
                [sys.executable, "-c", cmd],
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode == 0:
                output_data = json.loads(result.stdout.strip().split("\n")[-1])
                best_assignment = output_data["best_assignment"]
                best_coverage_time = output_data["best_coverage_time"]

                if best_coverage_time > 0:
                    if retry > 0:
                        print(
                            f"  FY{drone_num}->{missile_target}第{retry+1}次尝试成功，遮蔽时间: {best_coverage_time:.4f}s"
                        )
                    return best_assignment, best_coverage_time
                else:
                    print(
                        f"  FY{drone_num}->{missile_target}第{retry+1}次尝试遮蔽时间为0，重试中..."
                    )

        except Exception as e:
            print(f"  FY{drone_num}->{missile_target}第{retry+1}次尝试失败: {e}")

    print(f"FY{drone_num}->{missile_target}经过{max_retries}次尝试仍未找到有效解")
    return None, 0


def greedy_assign_drones_to_missiles():
    """
    使用贪心策略为无人机分配导弹目标

    Returns:
        list: 最优分配方案列表
    """
    print("开始贪心分配策略...")

    # 获取适配性矩阵
    priority_matrix = get_drone_missile_priority_matrix()

    # 存储最终分配结果
    final_assignments = []
    assigned_drones = set()
    missile_coverage = {"M1": [], "M2": [], "M3": []}  # 记录每个导弹的遮蔽区间

    # 可用的无人机和导弹
    available_drones = list(range(1, 6))  # FY1-FY5
    missiles = ["M1", "M2", "M3"]

    # 贪心分配过程
    for round_num in range(5):  # 最多5轮，每轮选择一架无人机
        if not available_drones:
            break

        print(f"\n=== 第{round_num + 1}轮贪心分配 ===")

        best_drone = None
        best_missile = None
        best_assignment = None
        best_increment = 0

        # 尝试每个可用无人机对每个导弹的组合
        for drone_num in available_drones:
            for missile in missiles:
                print(f"正在优化 FY{drone_num} -> {missile}...")

                # 将当前导弹的已有遮蔽区间转换为字符串格式
                excluded_intervals_data = None
                if missile_coverage[missile]:
                    excluded_intervals_data = [
                        str(interval) for interval in missile_coverage[missile]
                    ]

                # PSO优化
                assignment, coverage_time = optimize_drone_for_missile_pso(
                    drone_num, missile, excluded_intervals_data
                )

                if assignment and coverage_time > best_increment:
                    best_drone = drone_num
                    best_missile = missile
                    best_assignment = assignment
                    best_increment = coverage_time
                    print(
                        f"  找到更优方案: FY{drone_num} -> {missile}, 增量遮蔽: {coverage_time:.4f}s"
                    )

        # 选择最优分配
        if best_assignment:
            final_assignments.append(best_assignment)
            available_drones.remove(best_drone)

            # 计算并记录遮蔽区间
            interval = single_drone_multi_smoke_mask_time(best_assignment)
            missile_coverage[best_missile].append(interval)

            print(f"选择: FY{best_drone} -> {best_missile}")
            print(
                f"飞行参数: vx={best_assignment['flight_params'][0]:.2f}, vy={best_assignment['flight_params'][1]:.2f}"
            )
            print(f"烟幕弹数量: {len(best_assignment['smoke_bombs'])}")
            print(f"新增遮蔽时间: {best_increment:.4f}s")

        else:
            print("未找到有效的分配方案")
            break

    return final_assignments


def calculate_final_results(assignments):
    """
    计算最终的遮蔽效果

    Args:
        assignments: 无人机分配列表

    Returns:
        dict: 最终结果统计
    """
    results = calculate_total_mask_effectiveness(assignments)

    print("\n=== 最终遮蔽效果 ===")
    for missile in ["M1", "M2", "M3"]:
        print(f"{missile}: {results[missile]:.4f}s")
    print(f"总遮蔽时间: {results['total']:.4f}s")

    return results


def export_results_to_excel(assignments, filename="run/result3.xlsx"):
    """
    导出结果到Excel文件

    Args:
        assignments: 无人机分配列表
        filename: 输出文件名
    """
    # 准备数据
    export_data = []

    for assignment in assignments:
        drone_name = f"FY{assignment['drone_num']}"
        missile_target = assignment["missile_target"]
        vx, vy = assignment["flight_params"]

        for i, smoke in enumerate(assignment["smoke_bombs"]):
            export_data.append(
                {
                    "无人机": drone_name,
                    "目标导弹": missile_target,
                    "飞行速度x": vx,
                    "飞行速度y": vy,
                    "烟幕弹序号": i + 1,
                    "投放时间": smoke["drop_t"],
                    "起爆时间": smoke["bomb_t"],
                }
            )

    # 创建DataFrame并导出
    df = pd.DataFrame(export_data)
    df.to_excel(filename, index=False)
    print(f"\n结果已导出到: {filename}")


def main():
    """
    Q5主程序
    """
    print("=== Q5: 多无人机多导弹烟幕投放策略优化 ===\n")

    # 贪心分配
    assignments = greedy_assign_drones_to_missiles()

    if not assignments:
        print("未找到有效的分配方案！")
        return

    # 计算最终结果
    results = calculate_final_results(assignments)

    # 导出结果
    export_results_to_excel(assignments)

    print(f"\n优化完成！共分配了{len(assignments)}架无人机")
    print(f"总遮蔽效果: {results['total']:.4f}s")


if __name__ == "__main__":
    main()
