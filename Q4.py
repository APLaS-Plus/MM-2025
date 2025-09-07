import subprocess
import sys
import json
import math
import pandas as pd
import numpy as np
import random
from utils.Q4cal_mask_time import simgle_f_cal_mask_time
from utils.base import *
from utils.geo import *

# 计算参数
max_time = math.sqrt(sum(MISSILES_INITIAL["M1"] ** 2)) / MISSILE_SPEED


def optimize_single_drone_subprocess(
    drone_num,
    excluded_intervals_data=None,
    max_retries=10,
    custom_lb=None,
    custom_ub=None,
):
    """
    使用subprocess运行单个无人机的PSO优化，避免进程池累积
    支持重试机制克服PSO随机性问题

    Args:
        drone_num: 无人机编号 (1, 2, 3)
        excluded_intervals_data: 已被其他无人机占用的时间区间数据，格式为intervals的字符串表示
        max_retries: 最大重试次数
        custom_lb: 自定义下界 [vx_min, vy_min, drop_t_min, bomb_t_min]，None则使用默认值
        custom_ub: 自定义上界 [vx_max, vy_max, drop_t_max, bomb_t_max]，None则使用默认值

    Returns:
        (best_params, best_coverage_time, coverage_interval_data)
    """

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
from utils.Q4cal_mask_time import simgle_f_cal_mask_time
from utils.base import *
from utils.geo import *

# 计算参数
max_time = math.sqrt(sum(MISSILES_INITIAL["M1"] ** 2)) / MISSILE_SPEED
drone_num = {drone_num}
excluded_intervals_data = {repr(excluded_intervals_data)}
custom_lb = {custom_lb}
custom_ub = {custom_ub}

# 解析已有区间数据
excluded_intervals = None
if excluded_intervals_data:
    excluded_intervals = []
    for interval_str in excluded_intervals_data:
        # 重新构造sympy区间对象
        exec(f"from sympy import *")
        interval_obj = eval(interval_str)
        excluded_intervals.append(interval_obj)

def func(x):
    vx, vy, drop_t, bomb_t = x
    
    # 计算当前无人机的遮蔽区间
    interval = simgle_f_cal_mask_time((drone_num, vx, vy, drop_t, bomb_t))
    
    if excluded_intervals is None or len(excluded_intervals) == 0:
        # 第一台无人机，直接返回遮蔽时间
        coverage_time = interval.measure
    else:
        # 计算与已有区间的并集
        combined_interval = interval
        for exist_interval in excluded_intervals:
            combined_interval = combined_interval.union(exist_interval)
        
        # 新增的遮蔽时间 = 总遮蔽时间 - 原有遮蔽时间
        original_time = sum(intv.measure for intv in excluded_intervals)
        total_time = combined_interval.measure
        coverage_time = total_time - original_time
    
    return -coverage_time  # PSO求最小值，所以取负

# 约束条件：根据原始Q4约束逻辑修正
def constraint(x):
    vx, vy, drop_t, bomb_t = x
    # 速度约束：70² <= vx² + vy² <= 140² (参考原始vec_ueq)
    v_magnitude_sq = vx**2 + vy**2
    if not (70.0**2 <= v_magnitude_sq <= 140.0**2):
        return 1  # 违反约束
    # 时间约束：drop_t <= bomb_t and drop_t + bomb_t <= max_time (参考原始time_ueq)
    if not (drop_t <= bomb_t and drop_t + bomb_t <= max_time):
        return 1  # 违反约束
    return -1  # 满足约束

# 边界 - 根据自定义参数或默认约束条件设置
if custom_lb is not None:
    lb = custom_lb
else:
    lb = [-140, -140, 0, 0]

if custom_ub is not None:
    ub = custom_ub  
else:
    ub = [140, 140, max_time, max_time]
ueq = (constraint,)

w = 0.9
c = (1 - w) / 2

# PSO优化 - 使用多进程加速
set_run_mode(func, "multiprocessing")
pso = PSO(func=func, n_dim=4, pop=96, max_iter=200, lb=lb, ub=ub, constraint_ueq=ueq, c1=c, c2=c, w=w)
pso.run()

best_params = pso.gbest_x
best_coverage_time = -pso.gbest_y.item()

# 计算最优参数对应的遮蔽区间
vx, vy, drop_t, bomb_t = best_params
coverage_interval = simgle_f_cal_mask_time((drone_num, vx, vy, drop_t, bomb_t))

# 输出结果
result = {{
    "best_params": best_params.tolist(),
    "best_coverage_time": float(best_coverage_time),
    "coverage_interval_str": str(coverage_interval)
}}

print(json.dumps(result))
"""

        # 运行subprocess
        result = subprocess.run(
            [sys.executable, "-c", cmd],
            capture_output=True,
            text=True,
            timeout=300,  # 5分钟超时
        )

        if result.returncode == 0:
            try:
                output_data = json.loads(result.stdout.strip().split("\n")[-1])
                best_params = np.array(output_data["best_params"])
                best_coverage_time = output_data["best_coverage_time"]
                coverage_interval_str = output_data["coverage_interval_str"]

                # 重新构造sympy区间对象
                import sympy as sp
                from sympy import Interval, EmptySet, Union, oo

                coverage_interval = eval(coverage_interval_str)

                # 检查是否找到有效解
                if best_coverage_time > 0:
                    if retry > 0:  # 如果不是第一次尝试才显示重试信息
                        print(
                            f"  FY{drone_num}第{retry+1}次尝试成功，遮蔽时间: {best_coverage_time:.4f}s"
                        )
                    return best_params, best_coverage_time, coverage_interval
                else:
                    print(f"  FY{drone_num}第{retry+1}次尝试遮蔽时间为0，重试中...")

            except Exception as e:
                print(f"  FY{drone_num}第{retry+1}次尝试解析失败: {e}")
        else:
            print(f"  FY{drone_num}第{retry+1}次尝试subprocess失败")

    # 所有重试都失败了
    print(f"FY{drone_num}经过{max_retries}次尝试仍未找到有效解")
    return None, 0, None


def get_drone_search_bounds(drone_num):
    """
    根据无人机编号获取合适的搜索边界
    基于几何位置分析：导弹M1从(20000,0,2000)飞向(0,0,0)，真目标在(0,200,0)

    Args:
        drone_num: 无人机编号 (1, 2, 3)

    Returns:
        (custom_lb, custom_ub): 自定义的下界和上界 [vx_min, vy_min, drop_t_min, bomb_t_min]
    """
    max_time = math.sqrt(sum(MISSILES_INITIAL["M1"] ** 2)) / MISSILE_SPEED

    if drone_num == 1:  # FY1位置: (17800, 0, 1800) - 在导弹路径上，相对灵活
        # FY1可以各个方向移动，但偏向左移更容易拦截
        custom_lb = [-140, -30, 0, 0]
        custom_ub = [140, 30, max_time, max_time]
    elif drone_num == 2:  # FY2位置: (12000, 1400, 1400) - 右上方
        # FY2需要向左下移动才能接近拦截路径，限制向右上移动
        custom_lb = [-140, -140, 0, 0]
        custom_ub = [50, 50, max_time, max_time]  # 允许少量右上移动但主要向左下
    else:  # FY3位置: (6000, -3000, 700) - 右下方
        # FY3需要向左上移动才能接近拦截路径，限制向右下移动
        custom_lb = [-140, -50, 0, 0]  # 允许少量向下但主要向上
        custom_ub = [50, 140, max_time, max_time]

    return custom_lb, custom_ub


def greedy_optimize_q4():
    """
    使用贪心策略优化Q4问题
    """
    print("开始贪心优化...")

    # 存储结果
    selected_drones = []
    selected_params = []
    selected_intervals = []
    total_coverage = 0

    remaining_drones = [1, 2, 3]

    # 贪心选择过程
    for round_num in range(3):
        print(f"\n=== 第{round_num + 1}轮贪心选择 ===")

        best_drone = None
        best_params = None
        best_increment = 0
        best_interval = None

        # 为每台剩余无人机计算最优方案
        for drone_num in remaining_drones:
            print(f"正在优化无人机FY{drone_num}...")

            # 将已有区间转换为字符串格式，方便subprocess传递
            excluded_intervals_data = None
            if selected_intervals:
                excluded_intervals_data = [
                    str(interval) for interval in selected_intervals
                ]

            # 根据无人机位置和需求设置搜索边界
            custom_lb, custom_ub = get_drone_search_bounds(drone_num)

            # 可以在这里手动覆盖特定无人机的边界
            # 例如: 如果想让FY2只向左移动，可以设置:
            # if drone_num == 2: custom_ub[0] = 0  # x方向上界设为0

            params, coverage_time, interval = optimize_single_drone_subprocess(
                drone_num,
                excluded_intervals_data,
                custom_lb=custom_lb,
                custom_ub=custom_ub,
            )

            if params is not None:
                print(f"FY{drone_num}可提供额外遮蔽时间: {coverage_time:.4f}s")
            else:
                print(f"FY{drone_num}优化失败，跳过")
                coverage_time = 0

            if coverage_time > best_increment:
                best_drone = drone_num
                best_params = params
                best_increment = coverage_time
                best_interval = interval

        # 选择最优无人机
        if best_drone is not None:
            selected_drones.append(best_drone)
            selected_params.append(best_params)
            selected_intervals.append(best_interval)
            total_coverage += best_increment
            remaining_drones.remove(best_drone)

            print(
                f"选择FY{best_drone}，参数: vx={best_params[0]:.2f}, vy={best_params[1]:.2f}, "
                f"drop_t={best_params[2]:.2f}, bomb_t={best_params[3]:.2f}"
            )
            print(
                f"新增遮蔽时间: {best_increment:.4f}s，累计总遮蔽时间: {total_coverage:.4f}s"
            )

        if not remaining_drones:
            break

    return selected_drones, selected_params, total_coverage


# 主程序
if __name__ == "__main__":
    selected_drones, selected_params, total_coverage = greedy_optimize_q4()

    print(f"\n=== 最终结果 ===")
    print(f"选择顺序: {selected_drones}")
    print(f"总遮蔽时间: {total_coverage:.4f}s")

    for i, (drone_num, params) in enumerate(zip(selected_drones, selected_params)):
        vx, vy, drop_t, bomb_t = params
        print(
            f"FY{drone_num}: vx={vx:.2f}, vy={vy:.2f}, drop_t={drop_t:.2f}, bomb_t={bomb_t:.2f}"
        )

    # 保存结果
    results = []
    for drone_num, params in zip(selected_drones, selected_params):
        vx, vy, drop_t, bomb_t = params
        results.append(
            {
                "drone": f"FY{drone_num}",
                "vx": vx,
                "vy": vy,
                "drop_time": drop_t,
                "bomb_time": bomb_t,
            }
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv("run/Q4_greedy_results.csv", index=False)
    print(f"\n结果已保存到 run/Q4_greedy_results.csv")
