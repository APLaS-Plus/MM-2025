import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
import sys

# 添加上级目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.base import *
from utils.geo import *


def create_q1_animation():
    """创建Q1问题的飞行过程动画（x-z平面）"""

    # 设置中文字体
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 黑体
    plt.rcParams["axes.unicode_minus"] = False

    # 基础参数（从Q1.py复制）
    FY1_init_position = DRONES_INITIAL["FY1"].copy()
    FY_target = FAKE_TARGET.copy()
    FY_target[2] = FY1_init_position[2]  # 保持高度不变飞向假目标

    FY1_V = calculate_velocity_vector(FY1_init_position, FY_target, Q1_FY1_SPEED)

    M1_init_position = MISSILES_INITIAL["M1"].copy()
    M1_V = calculate_velocity_vector(M1_init_position, FAKE_TARGET, MISSILE_SPEED)

    # 计算关键时间点
    launch_time = Q1_LAUNCH_TIME  # 1.5s
    ignite_delay = Q1_IGNITE_INTERVAL  # 3.6s
    ignite_time = launch_time + ignite_delay  # 5.1s

    # 计算1.5s时FY1的位置（投放点）
    FY1_launch_pos = calculate_position_with_velocity(
        FY1_init_position, FY1_V, launch_time
    )

    # 计算烟雾弹起爆位置（5.1s时的抛物线位置）
    ignite_pos = calculate_parabolic_trajectory(FY1_launch_pos, FY1_V, ignite_delay)

    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))

    # 设置动画参数
    total_time = 20.0  # 总动画时间 20秒
    dt = 0.1  # 时间步长
    frames = int(total_time / dt)

    # 初始化线条和点
    (missile_line,) = ax.plot([], [], "r-", linewidth=2, label="M1导弹轨迹")
    (missile_point,) = ax.plot([], [], "ro", markersize=8, label="M1当前位置")

    (fy1_line,) = ax.plot([], [], "b--", linewidth=2, alpha=0.7, label="FY1无人机轨迹")
    (fy1_point,) = ax.plot([], [], "b^", markersize=8, label="FY1当前位置")

    (smoke_line,) = ax.plot([], [], "g:", linewidth=2, alpha=0.7, label="烟雾弹轨迹")
    (smoke_point,) = ax.plot([], [], "go", markersize=6, label="烟雾弹位置")

    # 烟雾云团（用圆圈表示）
    smoke_cloud = plt.Circle(
        (0, 0),
        SMOKE_EFFECTIVE_RADIUS,
        fill=False,
        color="gray",
        alpha=0.6,
        linewidth=3,
        linestyle="--",
        label="烟雾云团",
    )
    ax.add_patch(smoke_cloud)
    smoke_cloud.set_visible(False)

    # 用于存储轨迹点
    missile_trajectory_x = []
    missile_trajectory_z = []
    fy1_trajectory_x = []
    fy1_trajectory_z = []
    smoke_trajectory_x = []
    smoke_trajectory_z = []

    # 绘制固定目标（已移除，尺度太小）

    # 时间文本
    time_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # 状态文本
    status_text = ax.text(
        0.02,
        0.88,
        "",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    def animate(frame):
        current_time = frame * dt

        # 更新导弹位置
        if current_time <= 0:
            missile_pos = M1_init_position
        else:
            missile_pos = calculate_position_with_velocity(
                M1_init_position, M1_V, current_time
            )

        missile_trajectory_x.append(missile_pos[0])
        missile_trajectory_z.append(missile_pos[2])
        missile_line.set_data(missile_trajectory_x, missile_trajectory_z)
        missile_point.set_data([missile_pos[0]], [missile_pos[2]])

        # 更新FY1位置（仅在投放前显示）
        if current_time <= launch_time:
            fy1_pos = calculate_position_with_velocity(
                FY1_init_position, FY1_V, current_time
            )
            fy1_trajectory_x.append(fy1_pos[0])
            fy1_trajectory_z.append(fy1_pos[2])
            fy1_line.set_data(fy1_trajectory_x, fy1_trajectory_z)
            fy1_point.set_data([fy1_pos[0]], [fy1_pos[2]])
            fy1_point.set_visible(True)
        else:
            fy1_point.set_visible(False)

        # 更新烟雾弹/烟雾云团
        smoke_visible = False
        cloud_visible = False

        if launch_time < current_time <= ignite_time:
            # 烟雾弹抛物线飞行阶段
            smoke_time = current_time - launch_time
            smoke_pos = calculate_parabolic_trajectory(
                FY1_launch_pos, FY1_V, smoke_time
            )
            smoke_trajectory_x.append(smoke_pos[0])
            smoke_trajectory_z.append(smoke_pos[2])
            smoke_line.set_data(smoke_trajectory_x, smoke_trajectory_z)
            smoke_point.set_data([smoke_pos[0]], [smoke_pos[2]])
            smoke_visible = True

        elif current_time > ignite_time:
            # 烟雾云团下沉阶段
            time_after_ignite = current_time - ignite_time

            # 烟雾云团中心位置（垂直下沉）
            cloud_x = ignite_pos[0]
            cloud_z = ignite_pos[2] - SMOKE_SINK_SPEED * time_after_ignite

            # 更新烟雾云团位置
            smoke_cloud.center = (cloud_x, cloud_z)

            # 检查是否还在有效遮蔽时间内
            if time_after_ignite <= SMOKE_EFFECTIVE_TIME:
                cloud_visible = True
                smoke_cloud.set_alpha(0.6)
            else:
                cloud_visible = True
                smoke_cloud.set_alpha(0.3)  # 半透明显示失效的烟雾

        smoke_point.set_visible(smoke_visible)
        smoke_cloud.set_visible(cloud_visible)

        # 更新时间文本
        time_text.set_text(f"时间: {current_time:.1f} s")

        return (
            missile_line,
            missile_point,
            fy1_line,
            fy1_point,
            smoke_line,
            smoke_point,
            smoke_cloud,
            time_text,
            status_text,
        )

    # 计算导弹20秒后的位置，用作坐标轴参考
    missile_20s_pos = calculate_position_with_velocity(M1_init_position, M1_V, 20.0)

    # 设置坐标轴，让导弹20秒后位置作为左下角参考
    x_margin = 1000
    z_margin = 200
    ax.set_xlim(missile_20s_pos[0] - x_margin, M1_init_position[0] + x_margin)
    ax.set_ylim(missile_20s_pos[2] - z_margin, M1_init_position[2] + z_margin)
    ax.set_xlabel("X 坐标 (m)", fontsize=12)
    ax.set_ylabel("Z 坐标 (m)", fontsize=12)
    ax.set_title("Q1: FY1对M1烟雾干扰过程动画 (X-Z平面)", fontsize=14, weight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=10)

    # 创建动画
    anim = animation.FuncAnimation(
        fig, animate, frames=frames, interval=100, blit=False, repeat=True
    )

    # 保存动画（可选）
    save_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "run"
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 注释掉保存部分，因为需要ffmpeg
    save_path = os.path.join(save_dir, 'q1_animation.gif')
    anim.save(save_path, writer='pillow', fps=10)
    print(f'动画已保存至: {save_path}')

    plt.tight_layout()
    plt.show()

    return anim


if __name__ == "__main__":
    # 创建并显示动画
    anim = create_q1_animation()

    # 保持窗口打开
    plt.show()
