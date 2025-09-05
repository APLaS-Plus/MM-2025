# 几何计算相关函数

import numpy as np


def get_smoke_center_position(
    launch_pos, ignite_time, time_after_ignite, smoke_sink_speed=3, g=9.8
):
    """
    计算烟幕云团中心在起爆后某时刻的位置

    Args:
        launch_pos: 投放位置 np.array([x, y, z])
        ignite_time: 从投放到起爆的时间 (s)
        time_after_ignite: 起爆后的时间 (s)
        smoke_sink_speed: 烟幕下沉速度 (m/s)
        g: 重力加速度 (m/s^2)

    Returns:
        烟幕云团中心位置 np.array([x, y, z])
    """
    # 自由落体阶段
    fall_distance = 0.5 * g * ignite_time**2
    ignite_pos = launch_pos.copy()
    ignite_pos[2] -= fall_distance

    # 起爆后匀速下沉
    sink_distance = smoke_sink_speed * time_after_ignite
    current_pos = ignite_pos.copy()
    current_pos[2] -= sink_distance

    return current_pos


def calculate_distance_3d(pos1, pos2):
    """
    计算两点之间的三维距离

    Args:
        pos1: 第一个点的坐标 np.array([x, y, z])
        pos2: 第二个点的坐标 np.array([x, y, z])

    Returns:
        距离值 (m)
    """
    return np.linalg.norm(pos1 - pos2)


def point_to_line_distance_3d(point, line_point1, line_point2):
    """
    计算点到直线的最短距离（3D空间）

    Args:
        point: 点的坐标 np.array([x, y, z])
        line_point1: 直线上的第一个点 np.array([x1, y1, z1])
        line_point2: 直线上的第二个点 np.array([x2, y2, z2])

    Returns:
        distance: 点到直线的最短距离
        closest_point: 直线上距离最近的点
    """
    # 直线方向向量
    line_vec = line_point2 - line_point1
    point_vec = point - line_point1

    # 计算投影长度
    line_len = np.linalg.norm(line_vec)
    if line_len == 0:
        return np.linalg.norm(point - line_point1), line_point1

    line_unitvec = line_vec / line_len
    point_vec_scaled = point_vec / line_len

    t = np.dot(line_unitvec, point_vec_scaled)
    t = max(0.0, min(1.0, t))

    nearest = line_vec * t + line_point1
    distance = np.linalg.norm(point - nearest)

    return distance, nearest


def is_point_in_cylinder(point, cylinder_center, cylinder_radius, cylinder_height):
    """
    判断点是否在圆柱体内部

    Args:
        point: 点的坐标 np.array([x, y, z])
        cylinder_center: 圆柱体底面中心 np.array([cx, cy, cz])
        cylinder_radius: 圆柱体半径
        cylinder_height: 圆柱体高度

    Returns:
        bool: 点是否在圆柱体内部
    """
    # 检查高度
    if point[2] < cylinder_center[2] or point[2] > cylinder_center[2] + cylinder_height:
        return False

    # 检查水平距离
    horizontal_distance = np.sqrt(
        (point[0] - cylinder_center[0]) ** 2 + (point[1] - cylinder_center[1]) ** 2
    )

    return horizontal_distance <= cylinder_radius


def is_line_intersecting_sphere(point1, point2, sphere_center, sphere_radius):
    """
    判断两点之间的连线是否与球体相交（包括穿过球体或在球体内部）

    Args:
        point1: 第一个点的坐标 np.array([x1, y1, z1])
        point2: 第二个点的坐标 np.array([x2, y2, z2])
        sphere_center: 球心坐标 np.array([cx, cy, cz])
        sphere_radius: 球的半径

    Returns:
        bool: True 如果连线与球体相交，False 如果不相交
    """
    # 如果任一点在球内，则必定相交
    if np.linalg.norm(point1 - sphere_center) <= sphere_radius:
        return True
    if np.linalg.norm(point2 - sphere_center) <= sphere_radius:
        return True

    # 计算从 point1 到 point2 的方向向量
    line_vec = point2 - point1
    line_length = np.linalg.norm(line_vec)

    # 如果两点重合，且都在球外，则不相交
    if line_length == 0:
        return False

    # 归一化方向向量
    line_dir = line_vec / line_length

    # 计算从 point1 到球心的向量
    to_center = sphere_center - point1

    # 计算投影长度（球心在直线上的投影点到 point1 的距离）
    projection_length = np.dot(to_center, line_dir)

    # 如果投影点在线段外，检查端点到球心的距离
    if projection_length < 0:
        # 投影点在 point1 之前，最近点是 point1
        return np.linalg.norm(point1 - sphere_center) <= sphere_radius
    elif projection_length > line_length:
        # 投影点在 point2 之后，最近点是 point2
        return np.linalg.norm(point2 - sphere_center) <= sphere_radius

    # 投影点在线段上，计算投影点
    projection_point = point1 + projection_length * line_dir

    # 计算球心到直线的垂直距离
    distance_to_line = np.linalg.norm(sphere_center - projection_point)

    # 如果垂直距离小于等于球半径，则相交
    return distance_to_line <= sphere_radius


def calculate_velocity_vector(start_point, target_point, speed):
    """
    根据起始点、目标点和速度大小，计算速度向量

    Args:
        start_point: 起始点坐标 np.array([x1, y1, z1])
        target_point: 目标点坐标 np.array([x2, y2, z2])
        speed: 速度大小 (m/s)

    Returns:
        velocity_vector: 速度向量 np.array([vx, vy, vz])
        如果起始点和目标点重合，返回零向量
    """
    tp = target_point.copy()
    sp = start_point.copy()
    
    # 计算方向向量
    direction = tp - sp

    # 计算距离
    distance = np.linalg.norm(direction)

    # 如果起始点和目标点重合，返回零向量
    if distance == 0:
        return np.array([0.0, 0.0, 0.0])

    # 归一化方向向量
    unit_direction = direction / distance

    # 计算速度向量
    velocity_vector = unit_direction * speed

    return velocity_vector


def calculate_position_with_velocity(initial_position, velocity_vector, time):
    """
    根据初始位置、速度向量和时间，计算物体的当前位置（匀速直线运动）

    Args:
        initial_position: 初始位置坐标 np.array([x0, y0, z0])
        velocity_vector: 速度向量 np.array([vx, vy, vz]) (m/s)
        time: 经过的时间 (s)

    Returns:
        current_position: 当前位置坐标 np.array([x, y, z])
    """
    ip = initial_position.copy()
    vv = velocity_vector.copy()
    return ip + vv * time


def calculate_parabolic_trajectory(initial_position, initial_velocity, time, g=9.8):
    """
    根据初始位置、初始速度向量和时间，计算抛物线运动轨迹上的位置

    Args:
        initial_position: 初始位置坐标 np.array([x0, y0, z0])
        initial_velocity: 初始速度向量 np.array([vx, vy, vz]) (m/s)
        time: 经过的时间 (s)
        g: 重力加速度 (m/s^2)，默认为9.8

    Returns:
        current_position: 当前位置坐标 np.array([x, y, z])
    """
    # 水平方向匀速运动
    # 垂直方向受重力影响做匀加速运动
    ip = initial_position.copy()
    iv = initial_velocity.copy()
    current_position = ip.copy()
    current_position[0] = ip[0] + iv[0] * time  # x方向
    current_position[1] = ip[1] + iv[1] * time  # y方向
    current_position[2] = (
        ip[2] + iv[2] * time - 0.5 * g * time**2
    )  # z方向

    return current_position
