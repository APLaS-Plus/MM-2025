# 绘制无人机、目标物和导弹初始位置的3D图

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import sys

# 添加上级目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.base import (
    DRONES_INITIAL,
    MISSILES_INITIAL,
    FAKE_TARGET,
    TRUE_TARGET_CENTER,
    TRUE_TARGET_RADIUS,
    TRUE_TARGET_HEIGHT,
)


def plot_3d_positions():
    """绘制所有物体的初始位置"""
    # 创建3D图形
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # 设置中文字体
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 黑体
    plt.rcParams["axes.unicode_minus"] = False

    # 1. 绘制无人机位置
    for drone_name, pos in DRONES_INITIAL.items():
        ax.scatter(
            pos[0],
            pos[1],
            pos[2],
            c="blue",
            marker="^",
            s=100,
            label=f"{drone_name}" if drone_name == "FY1" else "",
        )
        ax.text(pos[0], pos[1], pos[2] + 100, drone_name, fontsize=8, ha="center")

    # 2. 绘制导弹位置
    for missile_name, pos in MISSILES_INITIAL.items():
        ax.scatter(
            pos[0],
            pos[1],
            pos[2],
            c="red",
            marker="v",
            s=100,
            label=f"导弹{missile_name[1]}" if missile_name == "M1" else "",
        )
        ax.text(pos[0], pos[1], pos[2] + 100, missile_name, fontsize=8, ha="center")

    # 3. 绘制假目标（原点）
    ax.scatter(
        FAKE_TARGET[0],
        FAKE_TARGET[1],
        FAKE_TARGET[2],
        c="orange",
        marker="*",
        s=200,
        label="假目标",
    )
    ax.text(
        FAKE_TARGET[0],
        FAKE_TARGET[1],
        FAKE_TARGET[2] + 100,
        "假目标",
        fontsize=10,
        ha="center",
        weight="bold",
    )

    # 4. 绘制真目标（圆柱体）
    # 圆柱体参数
    theta = np.linspace(0, 2 * np.pi, 30)
    z_cylinder = np.linspace(0, TRUE_TARGET_HEIGHT, 10)

    # 创建圆柱体网格
    theta_grid, z_grid = np.meshgrid(theta, z_cylinder)
    x_grid = TRUE_TARGET_RADIUS * np.cos(theta_grid) + TRUE_TARGET_CENTER[0]
    y_grid = TRUE_TARGET_RADIUS * np.sin(theta_grid) + TRUE_TARGET_CENTER[1]
    z_grid = z_grid + TRUE_TARGET_CENTER[2]

    # 绘制圆柱体表面
    ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.3, color="green")

    # 绘制圆柱体顶部和底部
    for z in [TRUE_TARGET_CENTER[2], TRUE_TARGET_CENTER[2] + TRUE_TARGET_HEIGHT]:
        x_circle = TRUE_TARGET_RADIUS * np.cos(theta) + TRUE_TARGET_CENTER[0]
        y_circle = TRUE_TARGET_RADIUS * np.sin(theta) + TRUE_TARGET_CENTER[1]
        z_circle = np.full_like(x_circle, z)
        ax.plot(x_circle, y_circle, z_circle, "g-", linewidth=2)

    # 标注真目标
    ax.text(
        TRUE_TARGET_CENTER[0],
        TRUE_TARGET_CENTER[1],
        TRUE_TARGET_CENTER[2] + TRUE_TARGET_HEIGHT + 50,
        "真目标",
        fontsize=10,
        ha="center",
        weight="bold",
        color="green",
    )

    # 5. 绘制导弹到假目标的轨迹线（虚线）
    for missile_name, pos in MISSILES_INITIAL.items():
        ax.plot(
            [pos[0], FAKE_TARGET[0]],
            [pos[1], FAKE_TARGET[1]],
            [pos[2], FAKE_TARGET[2]],
            "r--",
            alpha=0.3,
            linewidth=1,
        )

    # 设置图形属性
    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.set_zlabel("Z (m)", fontsize=12)
    ax.set_title("无人机、导弹和目标初始位置3D分布图", fontsize=16, weight="bold")

    # 设置视角
    ax.view_init(elev=20, azim=45)

    # 添加图例
    ax.legend(loc="upper right", fontsize=10)

    # 添加网格
    ax.grid(True, alpha=0.3)

    # 设置坐标轴范围
    ax.set_xlim(-1000, 21000)
    ax.set_ylim(-4000, 3000)
    ax.set_zlim(0, 2500)

    # 创建保存目录
    save_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "run"
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 保存图片
    save_path = os.path.join(save_dir, "initial_positions_3d.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"3D位置图已保存至: {save_path}")

    # 显示图形
    plt.show()

    return save_path


def plot_top_view():
    """绘制俯视图（XY平面）"""
    plt.figure(figsize=(10, 8))

    # 设置中文字体
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    # 绘制无人机
    for drone_name, pos in DRONES_INITIAL.items():
        plt.scatter(pos[0], pos[1], c="blue", marker="^", s=100)
        plt.text(
            pos[0] + 200, pos[1], f"{drone_name}\n({pos[2]}m)", fontsize=8, ha="left"
        )

    # 绘制导弹
    for missile_name, pos in MISSILES_INITIAL.items():
        plt.scatter(pos[0], pos[1], c="red", marker="v", s=100)
        plt.text(
            pos[0] + 200, pos[1], f"{missile_name}\n({pos[2]}m)", fontsize=8, ha="left"
        )

    # 绘制假目标
    plt.scatter(FAKE_TARGET[0], FAKE_TARGET[1], c="orange", marker="*", s=200)
    plt.text(
        FAKE_TARGET[0] + 200,
        FAKE_TARGET[1],
        "假目标",
        fontsize=10,
        ha="left",
        weight="bold",
    )

    # 绘制真目标
    circle = plt.Circle(
        (TRUE_TARGET_CENTER[0], TRUE_TARGET_CENTER[1]),
        TRUE_TARGET_RADIUS,
        fill=False,
        color="green",
        linewidth=2,
    )
    plt.gca().add_patch(circle)
    plt.text(
        TRUE_TARGET_CENTER[0],
        TRUE_TARGET_CENTER[1],
        "真目标",
        fontsize=10,
        ha="center",
        va="center",
        weight="bold",
        color="green",
    )

    # 绘制导弹轨迹
    for missile_name, pos in MISSILES_INITIAL.items():
        plt.plot(
            [pos[0], FAKE_TARGET[0]],
            [pos[1], FAKE_TARGET[1]],
            "r--",
            alpha=0.3,
            linewidth=1,
        )

    plt.xlabel("X (m)", fontsize=12)
    plt.ylabel("Y (m)", fontsize=12)
    plt.title("初始位置俯视图（XY平面）", fontsize=16, weight="bold")
    plt.grid(True, alpha=0.3)
    plt.axis("equal")

    # 保存图片
    save_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "run"
    )
    save_path = os.path.join(save_dir, "initial_positions_top_view.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"俯视图已保存至: {save_path}")

    plt.show()

    return save_path


if __name__ == "__main__":
    # 绘制3D图
    plot_3d_positions()

    # 绘制俯视图
    plot_top_view()
