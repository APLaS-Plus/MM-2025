import os
import sys
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

# ========== Q2优化结果参数 ==========
# 来自Q2.py的优化结果: Vx, drop, bomb: [-70.09965038   0.           2.46359669]
Vx_optimal = -70.09965038  # 无人机水平速度 (m/s)，负值表示向x轴负方向
drop_t_optimal = 0.0  # 投放时间 (s)
bomb_t_optimal = 2.46359669  # 起爆时间 (s)
mask_time_optimal = 3.131565782891446  # 最优遮蔽时间 (s)

# ========== 计算关键位置 ==========
# FY1初始位置和速度
FY1_init = DRONES_INITIAL["FY1"]
FY1_velocity = np.array([Vx_optimal, 0, 0])

# M1初始位置和速度
M1_init = MISSILES_INITIAL["M1"]
M1_velocity = calculate_velocity_vector(M1_init, FAKE_TARGET, MISSILE_SPEED)

# 投放位置（t=0立即投放，就是FY1的初始位置）
drop_position = FY1_init.copy()  # 立即投放，位置就是FY1初始位置

# 起爆位置
bomb_position = calculate_parabolic_trajectory(
    drop_position, FY1_velocity, bomb_t_optimal
)

# M1在起爆时的位置
M1_at_bomb_time = calculate_position_with_velocity(
    M1_init, M1_velocity, drop_t_optimal + bomb_t_optimal
)

# 真目标位置
true_target = TRUE_TARGET_CENTER

print(f"FY1初始位置: {FY1_init}")
print(f"投放位置: {drop_position}")
print(f"起爆位置: {bomb_position}")
print(f"M1起爆时位置: {M1_at_bomb_time}")
print(f"最优遮蔽时间: {mask_time_optimal:.3f}s")

# ========== 定义关键点坐标 (用于xz平面投影显示) ==========
points = {
    "O": np.array([0, 0]),  # 假目标(原点)
    "T": np.array([0, 0]),  # 真目标投影到xz平面
    "M1_init": np.array([M1_init[0], M1_init[2]]),  # M1初始位置
    "M1_bomb": np.array([M1_at_bomb_time[0], M1_at_bomb_time[2]]),  # M1起爆时位置
    "FY1_init": np.array([FY1_init[0], FY1_init[2]]),  # FY1初始位置
    "drop_pos": np.array([drop_position[0], drop_position[2]]),  # 投放位置
    "bomb_pos": np.array([bomb_position[0], bomb_position[2]]),  # 起爆位置
}

# 对应的3D实际坐标
coords_3d = {
    "O": "(0, 0, 0)",
    "T": "(0, 200, 0)",
    "M1_init": f"({M1_init[0]:.0f}, {M1_init[1]:.0f}, {M1_init[2]:.0f})",
    "M1_bomb": f"({M1_at_bomb_time[0]:.1f}, {M1_at_bomb_time[1]:.1f}, {M1_at_bomb_time[2]:.1f})",
    "FY1_init": f"({FY1_init[0]:.0f}, {FY1_init[1]:.0f}, {FY1_init[2]:.0f})",
    "drop_pos": f"({drop_position[0]:.0f}, {drop_position[1]:.0f}, {drop_position[2]:.0f})",
    "bomb_pos": f"({bomb_position[0]:.1f}, {bomb_position[1]:.1f}, {bomb_position[2]:.1f})",
}


# ========== 计算抛物线轨迹（从投放点到起爆点） ==========
def calculate_parabola_for_display(start_2d, end_2d, num_points=50):
    """计算抛物线轨迹点用于显示（基于实际物理计算的起爆位置）"""
    # 直接连接投放点和起爆点的抛物线轨迹
    x = np.linspace(start_2d[0], end_2d[0], num_points)

    # 使用简单的二次函数拟合投放点到起爆点
    x1, z1 = start_2d[0], start_2d[1]  # 投放点
    x2, z2 = end_2d[0], end_2d[1]  # 起爆点

    # 假设抛物线的顶点在中间，高度略高于起点
    x_mid = (x1 + x2) / 2
    z_mid = max(z1, z2) + abs(z1 - z2) * 0.1  # 稍微高一点形成抛物线形状

    # 使用三点拟合二次函数：z = ax² + bx + c
    # 通过投放点、中点、起爆点三点确定抛物线
    z_values = []
    for xi in x:
        t = (xi - x1) / (x2 - x1) if x2 != x1 else 0  # 参数化
        # 二次贝塞尔曲线插值
        zi = (1 - t) ** 2 * z1 + 2 * (1 - t) * t * z_mid + t**2 * z2
        z_values.append(zi)

    return np.column_stack((x, z_values))


# 计算抛物线轨迹
parabola_path = calculate_parabola_for_display(points["drop_pos"], points["bomb_pos"])

# ========== 创建绘图 ==========
fig, ax = plt.subplots(figsize=(15, 8))

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

# 2. M1运动轨迹（初始位置到起爆时位置）
ax.plot(
    [points["M1_init"][0], points["M1_bomb"][0]],
    [points["M1_init"][1], points["M1_bomb"][1]],
    "b--",
    linewidth=1.5,
    label="M1运动轨迹",
)

# 3. FY1轨迹（由于立即投放，显示为点）
ax.plot(
    [points["FY1_init"][0], points["drop_pos"][0]],
    [points["FY1_init"][1], points["drop_pos"][1]],
    "g-",
    linewidth=2,
    label="FY1飞行路径",
)

# 4. 干扰弹抛物线轨迹
ax.plot(
    parabola_path[:, 0],
    parabola_path[:, 1],
    "r--",
    linewidth=2,
    label="干扰弹抛物线轨迹",
)

# 5. 标记关键点
colors = ["red", "blue", "red", "blue", "green", "green", "orange"]
point_names = ["O", "T", "M1_init", "M1_bomb", "FY1_init", "drop_pos", "bomb_pos"]
for i, name in enumerate(point_names):
    if name == "T":
        # 真目标特殊处理，因为在xz平面上重合于原点
        continue
    coord = points[name]
    ax.scatter(*coord, s=50, c=colors[i], label=name, zorder=5)

    # 标签位置调整
    x_offset = 150 if coord[0] < 10000 else -150
    z_offset = 50 if name not in ["M1_bomb", "bomb_pos"] else -80

    ax.text(
        coord[0],
        coord[1] + 20,
        coords_3d[name],
        fontsize=9,
        ha="center",
    )

# 6. 绘制烟幕有效范围（应该与起爆点重合）
smoke_circle = Circle(
    (points["bomb_pos"][0], points["bomb_pos"][1]),  # 烟幕圆心应该与起爆点重合
    SMOKE_EFFECTIVE_RADIUS,
    color="green",
    fill=False,
    linestyle="-",
    linewidth=2,
    label=f"烟幕有效范围(R={SMOKE_EFFECTIVE_RADIUS}m)",
)
ax.add_patch(smoke_circle)


# ========== 添加优化结果文本 ==========
result_text = f"""Q2优化结果:
无人机速度: {abs(Vx_optimal):.1f} m/s (向X负方向)
投放时间: {drop_t_optimal:.1f} s (立即投放)
起爆时间: {bomb_t_optimal:.2f} s
最优遮蔽时长: {mask_time_optimal:.3f} s"""

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
ax.set_xlim(17000, 21000)
ax.set_ylim(1700, 2100)
ax.set_xlabel("X轴 (m)", fontsize=12)
ax.set_ylabel("Z轴 (m)", fontsize=12)
ax.set_title(
    "Q2最优烟幕干扰策略 - 导弹、无人机与干扰弹运动轨迹", fontsize=14, fontweight="bold"
)
ax.grid(True, linestyle="--", alpha=0.3)
ax.legend(loc="upper right", fontsize=9)

# 设置等比例坐标轴
# ax.set_aspect('equal', adjustable='box')

plt.tight_layout()

# # 保存图片
# save_dir = os.path.join(
#     os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "run"
# )
# save_path = os.path.join(save_dir, "Q2_anwser.png")
# plt.savefig(save_path, dpi=300, bbox_inches="tight")

plt.show()
