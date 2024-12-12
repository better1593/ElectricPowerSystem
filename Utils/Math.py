import numpy as np
import math
from tqdm import tqdm
from numba import jit
import time

def distance(pos1, pos2):
    return math.sqrt((pos1[0] - pos2[0]) ** 2 +
                     (pos1[1] - pos2[1]) ** 2 +
                     (pos1[2] - pos2[2]) ** 2)

def Bessel_K2(z, n1, n2):
    """
    【函数功能】修正的第二类Bessel函数相除Kn-1(z)/Kn(z)近似表达
    【入参】
    z(float): Bessel函数变量
    n1(int): 分子的Bessel函数阶数
    n2(int): 分母的Bessel函数阶数

    【出参】
    K2(float): Kn-1(z)/Kn(z)近似表达
    """
    K2 = (1 + ((4 * n1 ** 2 - 1) / (8 * z)) + ((4 * n1 ** 2 - 1) * (4 * n1 ** 2 - 9) / (2 * (8 * z) ** 2)) + (
            (4 * n1 ** 2 - 1) * (4 * n1 ** 2 - 9) * (4 * n1 ** 2 - 25) / (6 * (8 * z) ** 3))) / (
                 1 + ((4 * n2 ** 2 - 1) / (8 * z)) + (
                 (4 * n2 ** 2 - 1) * (4 * n2 ** 2 - 9) / (2 * (8 * z) ** 2)) + (
                         (4 * n2 ** 2 - 1) * (4 * n2 ** 2 - 9) * (4 * n2 ** 2 - 25) / (6 * (8 * z) ** 3)))
    return K2


def Bessel_IK(z1, n1, z2, n2):
    """
    【函数功能】修正的第一类Bessel函数和第二类Bessel函数相乘，In1(z1)*Kn2(z2)近似表达
    【入参】
    z1(float): 第一类Bessel函数变量
    n1(int): 第一类BBessel函数阶数
    z2(float): 第二类Bessel函数变量
    n2(int): 第二类Bessel函数阶数

    【出参】
    IK(float): In1(z1)*Kn2(z2)近似表达
    """
    IK = np.exp(z1 - z2) / 2 / np.sqrt(z1 * z2) * (
            1 - ((4 * n1 ** 2 - 1) / (8 * z1)) + ((4 * n2 ** 2 - 1) / (8 * z2)) - ((4 * n1 ** 2 - 1) / (8 * z1)) * (
            (4 * n2 ** 2 - 1) / (8 * z2)))
    return IK


def calculate_distances(points1, points2):
    """
    计算两个 n x 3 矩阵中对应行之间的距离.
    
    参数:
    points1 (np.ndarray): 第一个 n x 3 矩阵,每行表示一个点的 x, y, z 坐标
    points2 (np.ndarray): 第二个 n x 3 矩阵,每行表示一个点的 x, y, z 坐标
    
    返回:
    np.ndarray: 一个 n x 1 矩阵,表示两个矩阵中对应行之间的距离
    """
    if points1.shape != points2.shape:
        raise ValueError("两个输入矩阵必须有相同的形状!")
    
    distances = np.zeros((points1.shape[0], 1))
    for i in range(points1.shape[0]):
        distances[i] = np.sum((points1[i] - points2[i])**2)
    
    return distances


def calculate_direction_cosines(start_points, end_points, lengths):
    """
    计算 x、y 和 z 方向上的余弦值矩阵。

    参数:
    start_points (numpy.ndarray): n*3 矩阵,表示 n 条线段的起点坐标(x, y, z)
    end_points (numpy.ndarray): n*3 矩阵,表示 n 条线段的终点坐标(x, y, z)
    lengths (numpy.ndarray): n*1 矩阵,表示 n 条线段的长度

    返回:
    x_cosines, y_cosines, z_cosines(numpy.ndarray, numpy.ndarray, numpy.ndarray): x、y 和 z 方向上的余弦值矩阵
    """
    # 计算 x 方向上的余弦值
    x_cosines = (end_points[:, 0] - start_points[:, 0]).reshape(lengths.shape[0], 1) / lengths

    # 计算 y 方向上的余弦值
    y_cosines = (end_points[:, 1] - start_points[:, 1]).reshape(lengths.shape[0], 1) / lengths

    # 计算 z 方向上的余弦值
    z_cosines = (end_points[:, 2] - start_points[:, 2]).reshape(lengths.shape[0], 1) / lengths

    return x_cosines, y_cosines, z_cosines


def segment_branch(network_branches):
    """
    【功能】对于传进来的branches，按照读取的Nt进行分段
    """
    branches = network_branches.copy()  # branches的副本，用于新增或删减支路
    for key, value in network_branches.items():
        # key是支路， value是起点，终点，分段数
        keys_tobe_delete = []
        if 'OHL' in value[2]  and value[3] != 1:
            start_node_coord = list(value[0].values())[0]
            end_node_coord = list(value[1].values())[0]
            # 生成新节点
            x = np.linspace(start_node_coord[0], end_node_coord[0], value[3] + 1, dtype=float)
            y = np.linspace(start_node_coord[1], end_node_coord[1], value[3] + 1, dtype=float)
            z = np.linspace(start_node_coord[2], end_node_coord[2], value[3] + 1, dtype=float)
            new_nodes_name = [key + '_MiddleNode_{:02d}'.format(i) for i in range(1, value[3])]
            new_nodes_name.insert(0, list(value[0].keys())[0])
            new_nodes_name.append(list(value[1].keys())[0])
            for i in range(len(new_nodes_name) - 1):
                start_node_after_seg_dict = {new_nodes_name[i]: [x[i], y[i], z[i]]}
                end_node_after_seg_dict = {new_nodes_name[i+1]: [x[i+1], y[i+1], z[i+1]]}
                branches[f"{key}_splited_{i+1}"] = [start_node_after_seg_dict, end_node_after_seg_dict, value[2], value[3]]
            del branches[key]
    return branches

@jit(nopython=True)
def calculate_distances_between_lineseg_and_channelseg(points_a, points_b):
    """
    计算points_a中每个点到points_b中每个点的距离矩阵
    """
    # 计算差值
    differences_xyz = points_a[:, np.newaxis, :] - points_b[np.newaxis, :, :]
    differences_xy = points_a[:, np.newaxis, :2] - points_b[np.newaxis, :, :2]
    differences_z = points_a[:, np.newaxis, 2] - points_b[np.newaxis, :, 2]

    distances_xyz = np.sqrt(np.sum(differences_xyz ** 2, axis=2))
    distances_xy = np.sqrt(np.sum(differences_xy ** 2, axis=2))
    distances_z = differences_z
    distances_z.reshape(-1, 1)
    return distances_xyz, distances_xy, distances_z

@jit(nopython=True)
def get_t_delay_index1(arr):
    # 筛选出大于等于0的元素
    positive_part = arr[arr > 0]
    # 创建一个与输入数组相同长度的零数组
    index1 = np.zeros_like(arr)
    # 将大于0的元素放在前面
    index1[:len(positive_part)] = 1
    return index1

@jit(nopython=True)
def get_t_delay_index2(arr):
    # 筛选出大于等于0的元素
    positive_part = arr[arr > 0]
    # 创建一个与输入数组相同长度的零数组
    index2 = np.zeros_like(arr)
    # 将大于0的元素放在后面
    if positive_part.size > 0:
        index2[-len(positive_part):] = 1
    return index2


# def calculate_electric_field_down_r_and_z(pt_start, pt_end, stroke, channel, z_channel, i_sr, t_sr, i_sr_int, i_sr_div, ep0, vc, air_or_img):
#     """
#     功能：
#     计算雷击影响下，不同时刻，自由空间或镜像电场

#     参数说明：
#     pt_start (np.array, (n, 3)): 导体段的起点坐标，每行代表坐标(x, y, z)
#     pt_end (np.array, (n, 3)): 导体段的终点坐标，每行代表坐标(x, y, z)
#     stroke (Stroke对象)
#     channel (Channel对象)
#     z_channel (list): 每个雷电通道段终点的z坐标列表，从小到大排列
#     i_sr (np.array, (1, stroke.Nt): 雷电流时间序列
#     t_sr (np.array, (1, stroke.Nt): 时刻点
#     i_sr_int (np.array, (1, stroke.Nt): 雷电流时间序列的积分
#     i_sr_div (np.array, (1, stroke.Nt): 雷电流时间序列的微分
#     ep0 (float): 常数，真空介电常数
#     vc(float): 常数，光速
#     air_or_img (bool): 指示计算自由空间电场还是镜像电场

#     返回：
#     Ez (np.array, (stroke.Nt,  n): n为导体段的个数，z方向的电场，每行代表一个时刻点，每列代表导体段
#     Er (np.array, (stroke.Nt,  n): n为导体段的个数，r方向，即xoy平面方向的电场，每行代表一个时刻点，每列代表导体段
#     """
#     # 初始化径向和垂直方向的电场
#     a00 = pt_start.shape[0]  # 观察点或导体分段的总数
#     b00 = z_channel.shape[0]

#     line_mid_points = (pt_start + pt_end) / 2
#     channel_mid_points = np.column_stack( (np.full(z_channel.shape[0], channel.hit_pos[0]), np.full(z_channel.shape[0], channel.hit_pos[1]), z_channel))

#     # 空间距离和时间延迟的计算可以通过矩阵运算来实现
#     Rxyz, Rxy, Rz = calculate_distances_between_lineseg_and_channelseg(line_mid_points, channel_mid_points)

#     # 增加维度，便于广播
#     t_delay = (np.abs(z_channel) / vc / channel.vcof + Rxyz / vc)
#     t_delay_expand = np.tile(t_delay[:, :, np.newaxis], (1, 1, stroke.Nt))  # 变成 (a00, b00, 1)
#     t_sr_expand = np.tile(t_sr, (a00, b00, 1))  # 变成 (1, 1, 2000)

#     # 计算时间延迟，获取时间索引
#     # n_td_tmp = np.floor((t_sr_expand - t_delay_expand) / stroke.dt).astype(int)
#     n_td_tmp = np.floor((t_sr_expand * 1e-6 - t_delay_expand) / stroke.dt).astype(int)
#     index_head = np.apply_along_axis(get_t_delay_index1, 2, n_td_tmp)
#     index_tail = np.apply_along_axis(get_t_delay_index2, 2, n_td_tmp)
#     del n_td_tmp, t_sr_expand # 清内存
#     index_head = np.where(index_head == 1)
#     index_tail = np.where(index_tail == 1)
#     # id_t = n_td_tmp > 0


#     # 根据自由空间和镜像，选择对应的系数计算方式
#     if air_or_img == 0:
#         # 根据不同的传播模型，选择系数
#         if channel.channel_model == 'TL':
#             cof_isr = 1 / (4 * np.pi * ep0)
#         elif channel.channel_model == 'MTLL':
#             cof_isr = 1 / (4 * np.pi * ep0) * (1 - z_channel / channel.H)
#         else:
#             cof_isr = 1 / (4 * np.pi * ep0) * np.exp(-z_channel / channel.lamda)
#     else:
#         if channel.channel_model == 'TL':
#             cof_isr = 1 / (4 * np.pi * ep0)
#         elif channel.channel_model == 'MTLL':
#             cof_isr = 1 / (4 * np.pi * ep0) * (1 + z_channel / channel.H)
#         else:
#             cof_isr = 1 / (4 * np.pi * ep0) * np.exp(-np.abs(z_channel) / channel.lamda)

#     dEz_1_cof = cof_isr * (2 * Rz ** 2 - Rxy ** 2) / Rxyz ** 5
#     dEz_2_cof = cof_isr * (2 * Rz ** 2 - Rxy ** 2) / Rxyz ** 4 / vc
#     dEz_3_cof = cof_isr * (Rxy ** 2) / Rxyz ** 3 / vc ** 2

#     # 扩充维度为三维矩阵
#     dEz_1_cof = np.expand_dims(dEz_1_cof, axis=-1)
#     dEz_2_cof = np.expand_dims(dEz_2_cof, axis=-1)
#     dEz_3_cof = np.expand_dims(dEz_3_cof, axis=-1)

#     dEr_1_cof = cof_isr * (3 * Rz * Rxy) / Rxyz ** 5
#     dEr_2_cof = cof_isr * (3 * Rz * Rxy) / Rxyz ** 4 / vc
#     dEr_3_cof = cof_isr * (Rz * Rxy) / Rxyz ** 3 / vc ** 2
    
#     # 扩充维度为三维矩阵
#     dEr_1_cof = np.expand_dims(dEr_1_cof, axis=-1)
#     dEr_2_cof = np.expand_dims(dEr_2_cof, axis=-1)
#     dEr_3_cof = np.expand_dims(dEr_3_cof, axis=-1)

#     del Rxyz, Rz, Rxy

#     # 使用广播和布尔索引来累加对应时刻的贡献
#     i_sr_expand = np.tile(i_sr, (a00, b00, 1))
#     i_sr_int_expand = np.tile(i_sr_int, (a00, b00, 1))
#     i_sr_div_expand = np.tile(i_sr_div, (a00, b00, 1))

#     dEz1 = np.zeros((a00, b00, stroke.Nt))
#     dEz2 = np.zeros((a00, b00, stroke.Nt))
#     dEz3 = np.zeros((a00, b00, stroke.Nt))

#     dEz1[index_tail] = (dEz_1_cof * i_sr_int_expand)[index_head]
#     dEz2[index_tail] = (dEz_2_cof * i_sr_expand)[index_head]
#     dEz3[index_tail] = (dEz_3_cof * i_sr_div_expand)[index_head]

#     dEr1 = np.zeros((a00, b00, stroke.Nt))
#     dEr2 = np.zeros((a00, b00, stroke.Nt))
#     dEr3 = np.zeros((a00, b00, stroke.Nt))

#     dEr1[index_tail] = (dEr_1_cof * i_sr_int_expand)[index_head]
#     dEr2[index_tail] = (dEr_2_cof * i_sr_expand)[index_head]
#     dEr3[index_tail] = (dEr_3_cof * i_sr_div_expand)[index_head]

#     Ez = np.sum(dEz1 + dEz2 - dEz3, axis=1)
#     Er = np.sum(dEr1 + dEr2 + dEr3, axis=1)
#     # 转置电场矩阵
#     Ez = Ez.T
#     Er = Er.T
#     import psutil
#     import os
#     import time

#     process = psutil.Process(os.getpid())  # 获取当前进程 ID

#     mem_info = process.memory_info()
#     print(f"内存使用: {mem_info.rss / 1024 / 1024:.2f} MB")
#     return Ez, Er

def calculate_H_magnetic_field_down_r(pt_start, pt_end, stroke, channel, z_channel, i_sr, t_sr, i_sr_int, i_sr_div, ep0, vc, air_or_img):
    """
    功能：
    计算雷击影响下，不同时刻，r方向（z方向为0）的自由空间或镜像磁场？

    参数说明：
    pt_start (np.array, (n, 3)): 导体段的起点坐标，每行代表坐标(x, y, z)
    pt_end (np.array, (n, 3)): 导体段的终点坐标，每行代表坐标(x, y, z)
    stroke (Stroke对象)
    channel (Channel对象)
    z_channel (list): 每个雷电通道段终点的z坐标列表，从小到大排列
    i_sr (np.array, (1, stroke.Nt): 雷电流时间序列
    t_sr (np.array, (1, stroke.Nt): 时刻点
    i_sr_int (np.array, (1, stroke.Nt): 雷电流时间序列的积分
    i_sr_div (np.array, (1, stroke.Nt): 雷电流时间序列的微分
    ep0 (float): 常数，真空介电常数
    vc(float): 常数，光速
    air_or_img (bool): 指示计算自由空间电厂还是镜像电场

    返回：
    Ez (np.array, (stroke.Nt,  n): n为导体段的个数，每行代表一个时刻点，每列代表导体段
    Er (np.array, (stroke.Nt,  n): n为导体段的个数，每行代表一个时刻点，每列代表导体段
    """

    a00 = pt_start.shape[0]  # 观察点或导体分段的总数
    b00 = z_channel.shape[0]

    line_mid_points = (pt_start + pt_end) / 2
    channel_mid_points = np.column_stack( (np.full(z_channel.shape[0], channel.hit_pos[0]), np.full(z_channel.shape[0], channel.hit_pos[1]), z_channel))

    # 空间距离和时间延迟的计算可以通过矩阵运算来实现
    Rxyz, Rxy, Rz = calculate_distances_between_lineseg_and_channelseg(line_mid_points, channel_mid_points)

    # 增加维度，便于广播
    t_delay = (np.abs(z_channel) / vc / channel.vcof + Rxyz / vc)
    t_delay_expand = np.tile(t_delay[:, :, np.newaxis], (1, 1, stroke.Nt))  # 变成 (a00, b00, 1)
    t_sr_expand = np.tile(t_sr, (a00, b00, 1))  # 变成 (1, 1, 2000)

    # 计算时间延迟，获取时间索引
    # n_td_tmp = np.floor((t_sr_expand - t_delay_expand) / stroke.dt).astype(int)
    n_td_tmp = np.floor((t_sr_expand * 1e-6 - t_delay_expand) / stroke.dt).astype(int)
    index_head = np.apply_along_axis(get_t_delay_index1, 2, n_td_tmp)
    index_tail = np.apply_along_axis(get_t_delay_index2, 2, n_td_tmp)
    del t_delay_expand, n_td_tmp # 清除变量，释放内存
    index_head = np.where(index_head == 1)
    index_tail = np.where(index_tail == 1)
    # id_t = n_td_tmp > 0


    # 根据自由空间和镜像，选择对应的系数计算方式
    if air_or_img == 0:
        # 根据不同的传播模型，选择系数
        if channel.channel_model == 'TL':
            cof_isr = 1 / (4 * np.pi * ep0)
        elif channel.channel_model == 'MTLL':
            cof_isr = 1 / (4 * np.pi * ep0) * (1 - z_channel / channel.H)
        else:
            cof_isr = 1 / (4 * np.pi * ep0) * np.exp(-z_channel / channel.lamda)
    else:
        if channel.channel_model == 'TL':
            cof_isr = 1 / (4 * np.pi * ep0)
        elif channel.channel_model == 'MTLL':
            cof_isr = 1 / (4 * np.pi * ep0) * (1 + z_channel / channel.H)
        else:
            cof_isr = 1 / (4 * np.pi * ep0) * np.exp(-np.abs(z_channel) / channel.lamda)

    dEr_1_cof = 0 * cof_isr * (3 * Rz * Rxy) / Rxyz ** 5
    dEr_1_cof = np.expand_dims(dEr_1_cof, axis=-1)
    i_sr_int_expand = np.tile(i_sr_int, (a00, b00, 1))  # 使用广播和布尔索引来累加对应时刻的贡献
    dEr1 = np.zeros((a00, b00, stroke.Nt))
    dEr1[index_tail] = (dEr_1_cof * i_sr_int_expand)[index_head]
    del dEr_1_cof, i_sr_int_expand

    dEr_2_cof = cof_isr * (Rxy) / Rxyz ** 3
    dEr_2_cof = np.expand_dims(dEr_2_cof, axis=-1)
    i_sr_expand = np.tile(i_sr, (a00, b00, 1))
    dEr2 = np.zeros((a00, b00, stroke.Nt))
    dEr2[index_tail] = (dEr_2_cof * i_sr_expand)[index_head]
    del dEr_2_cof, i_sr_expand

    dEr_3_cof = cof_isr * (Rxy) / Rxyz ** 2 / vc
    dEr_3_cof = np.expand_dims(dEr_3_cof, axis=-1)
    i_sr_div_expand = np.tile(i_sr_div, (a00, b00, 1))
    dEr3 = np.zeros((a00, b00, stroke.Nt))
    dEr3[index_tail] = (dEr_3_cof * i_sr_div_expand)[index_head]
    del dEr_3_cof, i_sr_div_expand

    Er = np.sum(dEr1 + dEr2 + dEr3, axis=1)
    # 转置电场矩阵
    Er = Er.T
    # import psutil
    # import os
    # import time

    # process = psutil.Process(os.getpid())  # 获取当前进程 ID

    # mem_info = process.memory_info()
    # print(f"内存使用: {mem_info.rss / 1024 / 1024:.2f} MB")
    return Er


def calculate_electric_field_down_r_and_z(pt_start, pt_end, stroke, channel, z_channel, i_sr, t_sr, i_sr_int, i_sr_div, ep0, vc, air_or_img):
    # 初始化径向和垂直方向的电场
    a00 = pt_start.shape[0]  # 观察点或导体分段的总数
    b00 = z_channel.shape[0]

    # 雷击点位置
    hit_pos = channel.hit_pos[0:2]

    Rx = (pt_start[:, 0] + pt_end[:, 0]) / 2 - hit_pos[0]  # 雷击点在x方向到每个导线段中点的距离
    Ry = (pt_start[:, 1] + pt_end[:, 1]) / 2 - hit_pos[1]  # 雷击点在y方向到每个导线段中点的距离
    Rxy = np.sqrt(Rx ** 2 + Ry ** 2)  # 水平距离

    # 初始化径向和垂直方向的电场
    Er = np.zeros((stroke.Nt, a00))
    Ez = np.zeros((stroke.Nt, a00))

    for ik in range(a00):
        x1, y1, z1 = pt_start[ik, :]
        x2, y2, z2 = pt_end[ik, :]

        dEz1, dEz2, dEz3 = np.zeros((stroke.Nt, channel.N_channel_segment)), np.zeros((stroke.Nt, channel.N_channel_segment)), np.zeros((stroke.Nt, channel.N_channel_segment))
        dEr1, dEr2, dEr3 = np.zeros((stroke.Nt, channel.N_channel_segment)), np.zeros((stroke.Nt, channel.N_channel_segment)), np.zeros((stroke.Nt, channel.N_channel_segment))
        Rxyz = np.zeros(channel.N_channel_segment)
        Rz = np.zeros(channel.N_channel_segment)

        # 自由空间的电场计算
        for ig in range(channel.N_channel_segment):
            Rxyz[ig] = np.sqrt(Rxy[ik] ** 2 + ((z1 + z2) / 2 - z_channel[ig]) ** 2)  # 第ig个通道段与导体段的空间距离

            n_td_tmp = np.floor((t_sr * 1e-6 - (abs(z_channel[ig]) / vc / channel.vcof + Rxyz[ig] / vc)) / stroke.dt).astype(int)
            index_head = np.apply_along_axis(get_t_delay_index1, 1, n_td_tmp)
            index_tail = np.apply_along_axis(get_t_delay_index2, 1, n_td_tmp)
            index_head = np.where(index_head == 1)
            index_tail = np.where(index_tail == 1)
            id_t = n_td_tmp > 0

            Rz[ig] = (z1 + z2) / 2 - z_channel[ig]

            if air_or_img == 0:
                # 根据不同的传播模型，选择系数
                if channel.channel_model == 'TL':
                    cof_isr = 1 / (4 * np.pi * ep0)
                elif channel.channel_model == 'MTLL':
                    cof_isr = 1 / (4 * np.pi * ep0) * (1 - z_channel[ig] / channel.H)
                else:
                    cof_isr = 1 / (4 * np.pi * ep0) * np.exp(-z_channel[ig] / channel.lamda)
            else:
                if channel.channel_model == 'TL':
                    cof_isr = 1 / (4 * np.pi * ep0)
                elif channel.channel_model == 'MTLL':
                    cof_isr = 1 / (4 * np.pi * ep0) * (1 + z_channel[ig] / channel.H)
                else:
                    cof_isr = 1 / (4 * np.pi * ep0) * np.exp(-np.abs(z_channel[ig]) / channel.lamda)

            dEz_1_cof = cof_isr * (2 * Rz[ig] ** 2 - Rxy[ik] ** 2) / Rxyz[ig] ** 5
            dEz_2_cof = cof_isr * (2 * Rz[ig] ** 2 - Rxy[ik] ** 2) / Rxyz[ig] ** 4 / vc
            dEz_3_cof = cof_isr * Rxy[ik] ** 2 / Rxyz[ig] ** 3 / vc ** 2

            dEr_1_cof = cof_isr * 3 * Rz[ig] * Rxy[ik] / Rxyz[ig] ** 5
            dEr_2_cof = cof_isr * 3 * Rz[ig] * Rxy[ik] / Rxyz[ig] ** 4 / vc
            dEr_3_cof = cof_isr * Rz[ig] * Rxy[ik] / Rxyz[ig] ** 3 / vc ** 2

            dEz1[index_tail[1], ig] = (dEz_1_cof * i_sr_int)[index_head]
            dEz2[index_tail[1], ig] = (dEz_2_cof * i_sr)[index_head]
            dEz3[index_tail[1], ig] = (dEz_3_cof * i_sr_div)[index_head]

            dEr1[index_tail[1], ig] = (dEr_1_cof * i_sr_int)[index_head]
            dEr2[index_tail[1], ig] = (dEr_2_cof * i_sr)[index_head]
            dEr3[index_tail[1], ig] = (dEr_3_cof * i_sr_div)[index_head]

        Ez[:, ik] = np.sum(dEz1 + dEz2 - dEz3, axis=1)
        Er[:, ik] = np.sum(dEr1 + dEr2 + dEr3, axis=1)

    # # 转置电场矩阵
    # Ez = Ez.T
    # Er = Er.T

    return Ez, Er
#
#
# def calculate_H_magnetic_field_down_r(pt_start, pt_end, stroke, channel, z_channel, i_sr, t_sr, i_sr_int, i_sr_div, ep0, vc, air_or_img):
#     # 初始化径向和垂直方向的电场
#     a00 = pt_start.shape[0]  # 观察点或导体分段的总数
#     b00 = z_channel.shape[0]

#     # 雷击点位置
#     hit_pos = channel.hit_pos[0:2]


#     Rx = (pt_start[:, 0] + pt_end[:, 0]) / 2 - hit_pos[0]  # 雷击点在x方向到每个导线段中点的距离
#     Ry = (pt_start[:, 1] + pt_end[:, 1]) / 2 - hit_pos[1]  # 雷击点在y方向到每个导线段中点的距离
#     Rxy = np.sqrt(Rx ** 2 + Ry ** 2)  # 水平距离

#     # 初始化径向和垂直方向的电场
#     Er = np.zeros((stroke.Nt, a00))
#     Rxyz = np.zeros(channel.N_channel_segment)
#     Rz = np.zeros(channel.N_channel_segment)
#     Nt = stroke.Nt
#     N_channel_segment = channel.N_channel_segment
#     vcof = channel.vcof
#     dt = stroke.dt
#     channel_model = channel.channel_model
#     H = channel.H
#     lamda = channel.lamda
#     # @jit(nopython=True)
#     def loop(a00, pt_start, pt_end, Rxy, Rz, z_channel, t_sr, vc, vcof, dt, channel_model, H, lamda, ep0, i_sr_int, i_sr, i_sr_div, Nt, N_channel_segment, air_or_img, Er):
#         for ik in range(a00):
#             x1, y1, z1 = pt_start[ik, :]
#             x2, y2, z2 = pt_end[ik, :]

#             dEr1, dEr2, dEr3 = np.zeros((Nt, N_channel_segment)), np.zeros((Nt, N_channel_segment)), np.zeros((Nt, N_channel_segment))

#             Rxyz = np.sqrt(Rxy[ik] ** 2 + (z1 + z2) / 2 - z_channel) ** 2  # 第ig个通道段与导体段的空间距离

#             # 自由空间的电场计算
#             for ig in range(N_channel_segment):
#                 Rxyz[ig] = np.sqrt(Rxy[ik] ** 2 + ((z1 + z2) / 2 - z_channel[ig]) ** 2)  # 第ig个通道段与导体段的空间距离
#                 n_td_tmp = np.floor((t_sr * 1e-6 - (abs(z_channel[ig]) / vc / vcof + Rxyz[ig] / vc)) / dt).astype(int)
#                 # index_head = np.apply_along_axis(get_t_delay_index1, 1, n_td_tmp)
#                 # index_tail = np.apply_along_axis(get_t_delay_index2, 1, n_td_tmp)
#                 index_head = np.zeros(n_td_tmp.shape, dtype=int)
#                 index_tail = np.zeros(n_td_tmp.shape, dtype=int)
#                 for idx in range(n_td_tmp.shape[0]):
#                     arr = n_td_tmp[idx]
#                     # 筛选出大于等于0的元素
#                     positive_part = arr[arr > 0]
#                     # 创建一个与输入数组相同长度的零数组
#                     index1 = np.zeros_like(arr)
#                     # 将大于0的元素放在前面
#                     index1[:len(positive_part)] = 1
#                     index_head[idx] = index1

#                     arr = n_td_tmp[idx]
#                     # 筛选出大于等于0的元素
#                     positive_part = arr[arr > 0]
#                     # 创建一个与输入数组相同长度的零数组
#                     index2 = np.zeros_like(arr)
#                     # 将大于0的元素放在后面
#                     if positive_part.size > 0:
#                         index2[-len(positive_part):] = 1
#                     index_tail[idx] = index2
#                 index_head = np.where(index_head == 1)
#                 index_tail = np.where(index_tail == 1)

#                 Rz[ig] = (z1 + z2) / 2 - z_channel[ig]

#                 if air_or_img == 0:
#                     # 根据不同的传播模型，选择系数
#                     if channel_model == 'TL':
#                         cof_isr = 1 / (4 * np.pi * ep0)
#                     elif channel_model == 'MTLL':
#                         cof_isr = 1 / (4 * np.pi * ep0) * (1 - z_channel[ig] / H)
#                     else:
#                         cof_isr = 1 / (4 * np.pi * ep0) * np.exp(-z_channel[ig] / lamda)
#                 else:
#                     if channel_model == 'TL':
#                         cof_isr = 1 / (4 * np.pi * ep0)
#                     elif channel_model == 'MTLL':
#                         cof_isr = 1 / (4 * np.pi * ep0) * (1 + z_channel[ig] / H)
#                     else:
#                         cof_isr = 1 / (4 * np.pi * ep0) * np.exp(-np.abs(z_channel[ig]) / lamda)

#                 dEr_1_cof = 0 * cof_isr * 3 * Rz[ig] * Rxy[ik] / Rxyz[ig] ** 5
#                 dEr_2_cof = cof_isr * Rxy[ik] / Rxyz[ig] ** 3
#                 dEr_3_cof = cof_isr * Rxy[ik] / Rxyz[ig] ** 2 / vc
#                 # for x in range(len(index_tail[1])):
#                 #     dEr1[index_tail[1][x], ig] = dEr_1_cof * i_sr_int[index_head[0][x], index_head[1][x]]
#                 #     dEr2[index_tail[1][x], ig] = dEr_2_cof * i_sr[index_head[0][x], index_head[1][x]]
#                 #     dEr3[index_tail[1][x], ig] = dEr_3_cof * i_sr_div[index_head[0][x], index_head[1][x]]

#                 dEr1[index_tail[1], ig] = dEr_1_cof * i_sr_int[index_head]
#                 dEr2[index_tail[1], ig] = dEr_2_cof * i_sr[index_head]
#                 dEr3[index_tail[1], ig] = dEr_3_cof * i_sr_div[index_head]

#             Er[:, ik] = np.sum(dEr1 + dEr2 + dEr3, axis=1)
#         return Er
#     start_time = time.time()
#     Er = loop(a00, pt_start, pt_end, Rxy, Rz, z_channel, t_sr, vc, vcof, dt, channel_model, H, lamda, ep0, i_sr_int, i_sr, i_sr_div, Nt, N_channel_segment, air_or_img, Er)
#     # end_time = time.time()

#     # execution_time = end_time - start_time
#     # print(f"磁场计算时间Execution time: {execution_time} seconds")    # # 转置电场矩阵
#     # Er = Er.T
#     return Er


