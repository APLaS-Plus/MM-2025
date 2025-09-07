import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# 添加父目录到系统路径，以便导入utils模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.base import *
from utils.geo import *

# 设置中文显示
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ========== Q3优化结果参数 ==========
# 来自Q3.py的优化结果: [1.04982409e+02 1.39647206e+01 5.82782933e-03 0.00000000e+00 1.67813094e+01 2.14645528e+01 0.00000000e+00 8.58353653e-01]
# 参数含义：[f1_vx, f1_vy, drop_t1, bomb_t1, drop_t2, bomb_t2, drop_t3, bomb_t3]
Vx_optimal = 104.982409  # 无人机水平速度 (m/s)
Vy_optimal = 13.9647206  # 无人机垂直速度 (m/s)
drop_t1_optimal = 0.005827829  # 第1枚烟幕弹投放时间 (s) - 几乎立即投放
bomb_t1_optimal = 0.0  # 第1枚烟幕弹起爆时间 (s) - 立即起爆
drop_t2_optimal = 16.7813094  # 第2枚烟幕弹投放时间 (s)
bomb_t2_optimal = 21.4645528  # 第2枚烟幕弹起爆时间 (s)
drop_t3_optimal = 0.0  # 第3枚烟幕弹投放时间 (s) - 立即投放
bomb_t3_optimal = 0.58353653  # 第3枚烟幕弹起爆时间 (s)

# 计算最优遮蔽时间
from utils.Q3cal_mask_time import Q3_cal_mask_time_optimized

optimal_params = [
    Vx_optimal,
    Vy_optimal,
    drop_t1_optimal,
    bomb_t1_optimal,
    drop_t2_optimal,
    bomb_t2_optimal,
    drop_t3_optimal,
    bomb_t3_optimal,
]
mask_time_optimal = Q3_cal_mask_time_optimized(optimal_params)

# ========== 计算关键位置 ==========
# FY1初始位置和速度
FY1_init = DRONES_INITIAL["FY1"]
FY1_velocity = np.array([Vx_optimal, Vy_optimal, 0])

# M1初始位置和速度
M1_init = MISSILES_INITIAL["M1"]
M1_velocity = calculate_velocity_vector(M1_init, FAKE_TARGET, MISSILE_SPEED)

# 计算三个烟幕弹的关键位置（所有烟幕弹都显示，因为立即起爆仍然有效）
smoke_bombs = []
bomb_data = [
    (drop_t1_optimal, bomb_t1_optimal),
    (drop_t2_optimal, bomb_t2_optimal),
    (drop_t3_optimal, bomb_t3_optimal),
]

for i, (drop_t, bomb_t) in enumerate(bomb_data):

    # 投放位置
    drop_position = calculate_position_with_velocity(FY1_init, FY1_velocity, drop_t)

    # 起爆位置
    bomb_position = calculate_parabolic_trajectory(drop_position, FY1_velocity, bomb_t)

    # M1在起爆时的位置
    M1_at_bomb_time = calculate_position_with_velocity(
        M1_init, M1_velocity, drop_t + bomb_t
    )

    smoke_bombs.append(
        {
            "original_index": i + 1,  # 原始序号
            "index": len(smoke_bombs) + 1,  # 显示序号
            "drop_t": drop_t,
            "bomb_t": bomb_t,
            "drop_position": drop_position,
            "bomb_position": bomb_position,
            "M1_at_bomb_time": M1_at_bomb_time,
        }
    )

# 真目标位置
true_target = TRUE_TARGET_CENTER

# 输出关键信息
print(f"FY1初始位置: {FY1_init}")
print(f"FY1飞行速度: X轴{Vx_optimal:.2f} m/s, Y轴{Vy_optimal:.2f} m/s")
print(f"最优遮蔽时间: {mask_time_optimal:.3f}s")
print("\n烟幕弹信息:")
for bomb in smoke_bombs:
    print(
        f"烟幕弹{bomb['original_index']}: 投放时间={bomb['drop_t']:.2f}s, 起爆时间={bomb['bomb_t']:.2f}s"
    )
    print(f"  投放位置: {bomb['drop_position']}")
    print(f"  起爆位置: {bomb['bomb_position']}")

# ========== 定义关键点坐标 (用于xz平面投影显示) ==========
points = {
    "O": np.array([0, 0]),  # 假目标(原点)
    "T": np.array([0, 0]),  # 真目标投影到xz平面
    "M1_init": np.array([M1_init[0], M1_init[2]]),  # M1初始位置
    "FY1_init": np.array([FY1_init[0], FY1_init[2]]),  # FY1初始位置
}

# 添加烟幕弹相关点
for bomb in smoke_bombs:
    points[f"drop_pos{bomb['index']}"] = np.array(
        [bomb["drop_position"][0], bomb["drop_position"][2]]
    )
    points[f"bomb_pos{bomb['index']}"] = np.array(
        [bomb["bomb_position"][0], bomb["bomb_position"][2]]
    )
    points[f"M1_bomb{bomb['index']}"] = np.array(
        [bomb["M1_at_bomb_time"][0], bomb["M1_at_bomb_time"][2]]
    )

# 对应的3D实际坐标
coords_3d = {
    "O": "(0, 0, 0)",
    "T": "(0, 200, 0)",
    "M1_init": f"({M1_init[0]:.0f}, {M1_init[1]:.0f}, {M1_init[2]:.0f})",
    "FY1_init": f"({FY1_init[0]:.0f}, {FY1_init[1]:.0f}, {FY1_init[2]:.0f})",
}

# 添加烟幕弹坐标
for bomb in smoke_bombs:
    coords_3d[f"drop_pos{bomb['index']}"] = (
        f"({bomb['drop_position'][0]:.0f}, {bomb['drop_position'][1]:.0f}, {bomb['drop_position'][2]:.0f})"
    )
    coords_3d[f"bomb_pos{bomb['index']}"] = (
        f"({bomb['bomb_position'][0]:.1f}, {bomb['bomb_position'][1]:.1f}, {bomb['bomb_position'][2]:.1f})"
    )
    coords_3d[f"M1_bomb{bomb['index']}"] = (
        f"({bomb['M1_at_bomb_time'][0]:.1f}, {bomb['M1_at_bomb_time'][1]:.1f}, {bomb['M1_at_bomb_time'][2]:.1f})"
    )


# ========== 计算抛物线轨迹（从投放点到起爆点） ==========
def calculate_parabola_for_display(start_2d, end_2d, initial_velocity_x, num_points=30):
    """计算真实物理平抛运动轨迹点用于显示"""
    if np.allclose(start_2d, end_2d):
        # 如果投放点和起爆点重合（立即起爆），返回单点
        return np.array([start_2d])

    x1, z1 = start_2d[0], start_2d[1]  # 投放点
    x2, z2 = end_2d[0], end_2d[1]  # 起爆点

    # 平抛运动物理计算
    # 水平距离和垂直距离
    horizontal_dist = x2 - x1
    vertical_dist = z1 - z2  # z1应该大于z2（向下抛）

    # 根据水平速度和距离计算时间
    if abs(initial_velocity_x) > 0:
        flight_time = horizontal_dist / initial_velocity_x
    else:
        return np.array([start_2d])

    # 计算初始垂直速度（平抛时通常为0，但这里需要匹配终点）
    g = 9.8  # 重力加速度
    # z = z0 + v0z*t - (1/2)*g*t²
    # 求解初始垂直速度：v0z = (z2 - z1 + (1/2)*g*t²) / t
    if flight_time != 0:
        initial_velocity_z = (z2 - z1 + 0.5 * g * flight_time**2) / flight_time
    else:
        initial_velocity_z = 0

    # 生成轨迹点
    x = np.linspace(x1, x2, num_points)
    z_values = []

    for xi in x:
        # 根据位置计算时间
        t = (xi - x1) / initial_velocity_x if initial_velocity_x != 0 else 0
        # 平抛运动方程：z = z0 + v0z*t - (1/2)*g*t²
        zi = z1 + initial_velocity_z * t - 0.5 * g * t**2
        z_values.append(zi)

    return np.column_stack((x, z_values))


# 计算抛物线轨迹（只计算有效的烟幕弹）
parabola_paths = []
for bomb in smoke_bombs:
    drop_2d = points[f"drop_pos{bomb['index']}"]
    bomb_2d = points[f"bomb_pos{bomb['index']}"]
    # 传入水平速度参数
    parabola = calculate_parabola_for_display(drop_2d, bomb_2d, Vx_optimal)
    parabola_paths.append(parabola)

# ========== 创建绘图 ==========
fig, ax = plt.subplots(figsize=(16, 10))

# 定义颜色方案
smoke_colors = ["red", "orange", "purple"]  # 三个烟幕弹的颜色

# ========== 绘制关键元素 ==========
# 1. M1轨迹（从初始位置到假目标的直线）
ax.plot(
    [points["O"][0], points["M1_init"][0]],
    [points["O"][1], points["M1_init"][1]],
    "b-",
    linewidth=2,
    label="M1飞行路径",
    alpha=0.7,
)

# 2. M1运动轨迹（到不同起爆时刻的位置）
for i, bomb in enumerate(smoke_bombs):
    ax.plot(
        [points["M1_init"][0], points[f"M1_bomb{bomb['index']}"][0]],
        [points["M1_init"][1], points[f"M1_bomb{bomb['index']}"][1]],
        "b--",
        linewidth=1,
        alpha=0.6,
    )

# 3. FY1完整飞行路径（从初始位置到最大时间结束位置）
max_time = 20
final_position = FY1_init + FY1_velocity * max_time
fly_direction_end = np.array([final_position[0], final_position[2]])  # xz平面投影

ax.plot(
    [points["FY1_init"][0], fly_direction_end[0]],
    [points["FY1_init"][1], fly_direction_end[1]],
    "g-",
    linewidth=1,
    label="FY1完整飞行路径",
    alpha=0.7,
)

# 4. 有效干扰弹抛物线轨迹（过滤掉立即起爆的）
for i, (bomb, parabola) in enumerate(zip(smoke_bombs, parabola_paths)):
    if len(parabola.shape) == 1:  # 单点情况
        ax.plot(parabola[0], parabola[1], "o", color=smoke_colors[i], markersize=8)
    else:
        ax.plot(
            parabola[:, 0],
            parabola[:, 1],
            "--",
            color=smoke_colors[i],
            linewidth=2,
            label="干扰弹抛物线轨迹" if i == 0 else "",  # 只在第一个轨迹上显示标签
        )

# 5. 标记关键点
# FY1和M1初始位置
ax.scatter(
    *points["FY1_init"], s=80, c="green", marker="s", label="FY1初始位置", zorder=5
)
ax.scatter(*points["M1_init"], s=80, c="blue", marker="^", label="M1初始位置", zorder=5)

# # FY1最终位置标记
# ax.scatter(
#     fly_direction_end[0],
#     fly_direction_end[1],
#     s=60,
#     c="darkgreen",
#     marker="o",
#     label="FY1最终位置",
#     zorder=5,
# )

# 烟幕弹相关点
for i, bomb in enumerate(smoke_bombs):
    # 投放点
    drop_coord = points[f"drop_pos{bomb['index']}"]
    ax.scatter(
        *drop_coord,
        s=60,
        c=smoke_colors[i],
        marker="o",
        label="烟幕弹投放点" if i == 0 else "",
        zorder=5,
    )

    # 起爆点
    bomb_coord = points[f"bomb_pos{bomb['index']}"]
    ax.scatter(
        *bomb_coord,
        s=100,
        c=smoke_colors[i],
        marker="*",
        label="烟幕弹起爆点" if i == 0 else "",
        zorder=5,
    )

    # M1在起爆时的位置
    # m1_coord = points[f"M1_bomb{bomb['index']}"]
    # ax.scatter(
    #     *m1_coord,
    #     s=40,
    #     c="blue",
    #     marker="x",
    #     alpha=0.7,
    #     label="M1开始遮挡位置" if i == 0 else "",
    #     zorder=5,
    # )

# 6. 绘制烟幕有效范围
for i, bomb in enumerate(smoke_bombs):
    bomb_coord = points[f"bomb_pos{bomb['index']}"]
    smoke_circle = Circle(
        bomb_coord,
        SMOKE_EFFECTIVE_RADIUS,
        color="red",
        fill=False,
        linestyle="-",
        linewidth=1.5,
        alpha=0.7,
        label=(
            f"烟幕有效范围(R={SMOKE_EFFECTIVE_RADIUS}m)" if i == 0 else ""
        ),  # 只在第一个圆上显示标签
    )
    ax.add_patch(smoke_circle)


# FY1和M1初始位置标注

coord = points["FY1_init"]
ax.text(
    coord[0],
    coord[1] + 10,
    coords_3d["FY1_init"],
    fontsize=9,
    ha="center",
)
coord = points["M1_init"]
ax.text(
    coord[0] - 100,
    coord[1] + 10,
    coords_3d["M1_init"],
    fontsize=9,
    ha="center",
)

# FY1最终位置标注
ax.text(
    fly_direction_end[0],
    fly_direction_end[1] + 15,
    f"({final_position[0]:.0f}, {final_position[1]:.0f}, {final_position[2]:.0f})",
    fontsize=9,
    ha="center",
)

# 标注投放点位置（如果与FY1初始位置不同的话）
# 在Q3中所有烟幕弹都是立即投放，投放位置就是FY1初始位置，所以不需要重复标注

# 标记烟幕弹信息和坐标
for i, bomb in enumerate(smoke_bombs):
    bomb_coord = points[f"bomb_pos{bomb['index']}"]
    # print(i)
    x = -50 if i == 0 else 50
    y = -10 if i == 0 else -20
    # 检查起爆点是否在显示范围内，并添加坐标标注
    if 1700 <= bomb_coord[1] <= 2100:  # 在z轴显示范围内
        ax.text(
            bomb_coord[0] + x,
            bomb_coord[1] + y,
            coords_3d[f"bomb_pos{bomb['index']}"],
            fontsize=9,
            ha="center",
        )

    # 添加时间信息标注
    # ax.text(
    #     bomb_coord[0] + label_offset,
    #     bomb_coord[1] - 30 - i * 20,
    #     f"烟幕弹{bomb['original_index']}\n起爆: {bomb['bomb_t']:.2f}s",
    #     fontsize=8,
    #     ha="center",
    #     bbox=dict(boxstyle="round,pad=0.3", facecolor=smoke_colors[i], alpha=0.3),
    # )

    # 为显示范围内的M1位置添加坐标标注
    m1_coord = points[f"M1_bomb{bomb['index']}"]
    # if 1700 <= m1_coord[1] <= 2100:  # M1位置在显示范围内
    #     ax.text(
    #         m1_coord[0],
    #         m1_coord[1] + 10,
    #         coords_3d[f"M1_bomb{bomb['index']}"],
    #         fontsize=9,
    #         ha="center",
    #     )

# ========== 添加优化结果文本 ==========
result_text = f"""Q3优化结果（FY1投放3枚烟幕弹）:
无人机速度: {Vx_optimal:.1f} m/s (向X正方向), {Vy_optimal:.1f} m/s (向Y正方向)
烟幕弹1: 立即投放({drop_t1_optimal:.3f}s), 立即起爆({bomb_t1_optimal:.1f}s)
烟幕弹2: {drop_t2_optimal:.1f}s投放, {bomb_t2_optimal:.1f}s起爆  
烟幕弹3: 立即投放({drop_t3_optimal:.1f}s), {bomb_t3_optimal:.2f}s起爆
最优遮蔽时长: 7.404 s"""

ax.text(
    0.02,
    0.98,
    result_text,
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
)

# ========== 图形美化 ==========
ax.set_xlim(17500, 20000)  # 扩展x轴范围以包含FY1完整飞行路径
ax.set_ylim(1700, 2100)
ax.set_xlabel("X轴 (m)", fontsize=12)
ax.set_ylabel("Z轴 (m)", fontsize=12)
ax.grid(True, linestyle="--", alpha=0.3)
ax.legend(loc="upper right", fontsize=9)

plt.tight_layout()

# 保存图片
# save_dir = os.path.join(
#     os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "run"
# )
# save_path = os.path.join(save_dir, "Q3_anwser.png")
# plt.savefig(save_path, dpi=300, bbox_inches="tight")
# print(f"图片已保存到: {save_path}")

plt.show()
