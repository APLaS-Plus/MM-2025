import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# 设置中文显示
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ========== 定义关键点坐标 (xz平面) ==========
points = {
    "O": np.array([0, 0]),  # 原点
    "M": np.array([20000, 2000]),  # 导弹初始位置
    "M'": np.array([18477.59, 1847.76]),  # 导弹运动后位置
    "FY1": np.array([17800, 2000]),  # 无人机初始位置
    "FY1'": np.array([17188, 2000]),  # 无人机飞行后位置
    "投放点": np.array([17620, 2000]),  # 干扰弹投放点（新增）
    "S": np.array([17188, 1737.50]),  # 烟幕起爆点
}

# 对应的3D实际坐标
coords_3d = {
    "O": "(0, 0, 0)",
    "M": "(20000, 0, 2000)",
    "M'": "(18477.59, 0, 1847.76)",
    "FY1": "(17800, 0, 1800)",
    "FY1'": "(17188, 0, 1800)",
    "S": "(17188, 0, 1736.50)",
}


# ========== 计算抛物线轨迹（从投放点到起爆点） ==========
def calculate_parabola(start, end, num_points=50):
    """基于物理模型计算抛物线轨迹（考虑重力加速度）"""
    x = np.linspace(start[0], end[0], num_points)
    # 抛物线方程 z = z0 - (g/2v0²)(x - x0)²
    g = 9.8  # 重力加速度
    horiz_dist = end[0] - start[0]
    fall_height = start[1] - end[1]
    # 计算初速度v0（确保抛物线通过起爆点）
    v0 = np.sqrt(g * horiz_dist**2 / (2 * fall_height))
    z = start[1] - (g / (2 * v0**2)) * (x - start[0]) ** 2
    return np.column_stack((x, z))


# 计算从投放点(17620,2000)到S(17188,1737.50)的抛物线
parabola_path = calculate_parabola(points["投放点"], points["S"])

# ========== 创建绘图 ==========
fig, ax = plt.subplots(figsize=(14, 7))

# ========== 绘制关键元素 ==========
# 1. 导弹轨迹（OM和MM'）
ax.plot(
    [points["O"][0], points["M'"][0]],
    [points["O"][1], points["M'"][1]],
    "b-",
    linewidth=2,
    label="导弹原始路径OM",
)
ax.plot(
    [points["M"][0], points["M'"][0]],
    [points["M"][1], points["M'"][1]],
    "b--",
    linewidth=1.5,
    label="导弹运动MM'",
)

# 2. 无人机轨迹（FY1到FY1'）
ax.plot(
    [points["FY1"][0], points["FY1'"][0]],
    [points["FY1"][1], points["FY1'"][1]],
    "g-",
    linewidth=2,
    label="无人机飞行路径",
)

# 3. 干扰弹抛物线（从投放点到S）
ax.plot(
    parabola_path[:, 0],
    parabola_path[:, 1],
    "r--",
    linewidth=1.5,
    label="干扰弹抛物线轨迹",
)

# 4. 标记关键点
for name, coord in points.items():
    if name == "投放点":
        continue
    ax.scatter(*coord, s=80, label=name, zorder=5)
    # 动态调整标签位置
    x_offset = 100 if name not in ["M'", "FY1'", "S"] else -100
    z_offset = 20 if name != "S" else -40
    # 只显示3D坐标
    ax.text(coord[0] + x_offset, coord[1] + z_offset, coords_3d[name], fontsize=10)

# 5. 绘制烟幕有效范围（半径10m）
smoke_circle = Circle(
    (points["S"][0], points["S"][1]),
    10,
    color="orange",
    fill=False,
    linestyle=":",
    linewidth=1.5,
    label="烟幕有效范围",
)
ax.add_patch(smoke_circle)

# ========== 图形标注 ==========
ax.annotate(
    [17620, 0, 1800],
    xy=points["投放点"],
    xytext=(17700, 1950),
    arrowprops=dict(arrowstyle="->"),
    fontsize=10,
)
ax.annotate(
    coords_3d["S"],
    xy=points["S"],
    xytext=(17100, 1650),
    arrowprops=dict(arrowstyle="->"),
    fontsize=10,
)

# ========== 图形美化 ==========
ax.set_xlim(17000, 21000)
ax.set_ylim(1700, 2100)
ax.set_xlabel("X轴 (m)", fontsize=12)
ax.set_ylabel("Z轴 (m)", fontsize=12)
ax.set_title("导弹、无人机与干扰弹运动轨迹", fontsize=14)
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1))

plt.tight_layout()

# # 保存图片（必须在plt.show()之前）
# save_dir = os.path.join(
#     os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "run"
# )
# save_path = os.path.join(save_dir, "Q1_anwser.png")
# plt.savefig(save_path, dpi=300, bbox_inches="tight")

plt.show()
