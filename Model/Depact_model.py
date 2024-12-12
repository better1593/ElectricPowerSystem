from Function.Calculators.Inductance import *
from Function.Calculators.Capacitance import *
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from Contant import Constant
from Ground import Ground
from Function.Calculators.InducedVoltage_calculate import ElectricField_calculate, H_MagneticField_calculate, ElectricField_above_lossy
from Lightning import Stroke, Channel
import pandas as pd
from Vector_Fitting.Calculators.vecfit_kernel_z import vecfit_kernel_Z_Ding
from tqdm import tqdm

def write_to_txt(array, filepath):
    # Your NumPy array

    # Ensure the array has an even number of elements
    if array.size % 2 != 0:
        array = np.append(array, np.nan)  # Add a padding value if necessary

    # Reshape into two columns
    array_reshaped = array.reshape(-1, 2)

    # Save to TXT file with two columns
    np.savetxt(filepath, array_reshaped, fmt='%.20e', delimiter='\t', header='', comments='')

radius = np.array([9.14e-3, 9.14e-3, 9.14e-3])
height = np.array([17.4, 13.7, 10.0])
y = np.array([0, 0, 0])
constants = Constant()
ground = Ground(sig=0.001, epr=10)
mu0 = constants.mu0

L_matrix = calculate_OHL_mutual_inductance(radius, height, y, mu0)
C_matrix = calculate_OHL_capcitance(L_matrix, constants)

import numpy as np

def generate_transform_matrix(n):
    matrix = np.zeros((n, n))
    sqrt_n = np.sqrt(n)
    # 第一列元素全部为 1 / sqrt(n)
    matrix[:, 0] = 1 / sqrt_n

    # 设置对角线元素（除第一行第一列）
    for i in range(1, n):
        matrix[i, i] = -1 / np.sqrt((i + 1) * i)

    # 设置上对角线元素
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i, j] = 1 / np.sqrt((j + 1) * j)

    return matrix

# L_eigenvalues, L_eigenvectors = np.linalg.eig(L_matrix)
# C_eigenvalues, C_eigenvectors = np.linalg.eig(C_matrix)

Tv = generate_transform_matrix(L_matrix.shape[0])
Tv_inv = np.linalg.inv(Tv)
Ti = np.linalg.inv(Tv.T)
Ti_inv = Tv.T

def ensure_positive_definite(matrix, threshold=1e-20):
    m = np.zeros_like(matrix)
    # 将所有小于阈值的特征值设为阈值
    np.fill_diagonal(m, np.diag(matrix))
    return m
# 模态电感和电容
print(Tv_inv @ Ti)
print(Ti_inv @ Tv)
Lm_matrix = ensure_positive_definite(Tv_inv @ L_matrix @ Ti)
Cm_matrix = ensure_positive_definite(Ti_inv @ C_matrix @ Tv)
Zm_matrix = ensure_positive_definite(np.sqrt(np.divide(Lm_matrix, Cm_matrix)))
Zm_matrix_inv = np.linalg.inv(Zm_matrix)
Vm_matrix = ensure_positive_definite(1 / np.sqrt(Lm_matrix * Cm_matrix))
Vm_matrix = np.diag(Vm_matrix)

stroke1 = Stroke('Heidler', duration=1.0e-5, dt=1e-8, is_calculated=True, parameter_set='0.25/2.5us',
                parameters=None)
stroke1.calculate()
channel = Channel(hit_pos=[0, 0, 0])
plt.plot(stroke1.t_us, stroke1.current_waveform)
plt.show()
def divide_line_segment(start, end, num_segments):
    """
    根据起点和终点的三维坐标，以及均匀划分的段数，得到分段后的线段起点和终点三维坐标矩阵。

    参数:
    start (tuple): 线段起点的三维坐标 (x, y, z)
    end (tuple): 线段终点的三维坐标 (x, y, z)
    num_segments (int): 均匀划分的段数

    返回:
    tuple: (起点矩阵, 终点矩阵)
    """
    start = np.array(start)
    end = np.array(end)
    
    # 生成均匀分布的参数 t
    t = np.linspace(0, 1, num_segments + 1).reshape(-1, 1)
    
    # 计算分段后的坐标
    points = start + t * (end - start)
    
    # 起点矩阵和终点矩阵
    start_points = points[:-1]
    end_points = points[1:]
    
    return start_points, end_points
seg_seq = 0
seg_num = 25
x_start = 500
x_end = -500
dx = (x_start - x_end) / seg_num
pt_start1, pt_end1 = divide_line_segment((x_start, 50, 17.4), (x_end, 50, 17.4), seg_num)
pt_start2, pt_end2 = divide_line_segment((x_start, 50, 13.7), (x_end, 50, 13.7), seg_num)
pt_start3, pt_end3 = divide_line_segment((x_start, 50, 10), (x_end, 50, 10), seg_num)
pt_start = np.vstack((pt_start1, pt_start2, pt_start3))
pt_end = np.vstack((pt_end1, pt_end2, pt_end3))

Ez_T, Er_T = ElectricField_calculate(pt_start, pt_end, stroke1, channel, constants.ep0, constants.vc)  # 计算电场

H_p = H_MagneticField_calculate(pt_start, pt_end, stroke1, channel, constants.ep0, constants.vc)  # 计算磁场

# 计算有损地面的电场
Er_lossy = ElectricField_above_lossy(-H_p, Er_T, constants, constants.sigma)
Er_lossy = Er_T.T

plt.plot(stroke1.t_us, Er_lossy[0, :])
plt.plot(stroke1.t_us, Er_lossy[1, :])
plt.plot(stroke1.t_us, Er_lossy[2, :])
plt.show()
# x方向的入射电场计算
a00 = pt_start.shape[0]  # 导体段个数

Rx = (pt_start[:, 0] + pt_end[:, 0]) / 2 - channel.hit_pos[0]
Ry = (pt_start[:, 1] + pt_end[:, 1]) / 2 - channel.hit_pos[1]
Rxy = np.sqrt(Rx**2 + Ry**2)

Ex_lossy = np.zeros((a00, stroke1.Nt))  # 初始化矩阵

for ik in range(a00):
    x1, y1, z1 = pt_start[ik]
    x2, y2, z2 = pt_end[ik]

    # if Rxy[ik] == 0:
    #     U_fm1[:, ik] = Ez_lossy[:, ik] * (z1 - z2)
    # else:
    cosx = Rx[ik] / Rxy[ik]
    cosy = Ry[ik] / Rxy[ik]
    Ex_lossy[ik, :] = Er_lossy[ik, :] * cosx
print(Ex_lossy)

plt.plot(stroke1.t_us, Ex_lossy[0, :])
plt.plot(stroke1.t_us, Ex_lossy[1, :])
plt.plot(stroke1.t_us, Ex_lossy[2, :])
plt.show()

Ex_lossy1 = Ex_lossy[0:seg_num, :]
Ex_lossy2 = Ex_lossy[seg_num:2*seg_num, :]
Ex_lossy3 = Ex_lossy[2*seg_num:3*seg_num, :]
combined_matrix = np.stack((Ex_lossy1, Ex_lossy2, Ex_lossy3), axis=0)
for i in range(combined_matrix.shape[2]):
    combined_matrix[:, :, i] = Tv @ combined_matrix[:, :, i]
modal_Ex_lossy = combined_matrix
# plt.plot(stroke1.t_us, modal_Ex_lossy[0, 0, :])
# plt.plot(stroke1.t_us, modal_Ex_lossy[0, 1, :])
# plt.plot(stroke1.t_us, modal_Ex_lossy[0, 2, :])
# plt.show()
def delay_waveform(waveform, m):
    delayed_waveform = np.zeros_like(waveform)
    if m < len(waveform):
        delayed_waveform[m:] = waveform[:-m]
    return delayed_waveform

# 利用公式U = E * L计算积分
def integral(pt_start, pt_end, vm, stroke, channel, Ex, dx, flag):
    a00 = pt_start.shape[0]  # 导体段个数

    U = np.zeros((a00, stroke.Nt))  # 初始化矩阵
    if flag == 0:
        time_lag = np.ceil((np.abs((pt_start[:, 0] + pt_end[:, 0]) / 2 - pt_start[0, 0])) / vm / stroke.dt).astype(int)
    else:
        time_lag = np.ceil((np.abs((-pt_start[:, 0] + pt_end[:, 0]) / 2 + pt_end[-1, 0])) / vm / stroke.dt).astype(int)
    for ik in range(a00):
        Ex[ik, :] = delay_waveform(Ex[ik, :], time_lag[ik])
        U[ik, :] = Ex[ik, :] * dx
    return np.sum(U, axis=0)

modal_Ex_source1 = np.zeros((modal_Ex_lossy.shape[0], stroke1.Nt))
modal_Ex_source2 = np.zeros((modal_Ex_lossy.shape[0], stroke1.Nt))
for i in range(modal_Ex_lossy.shape[0]):
    modal_Ex_source1[i, :] = integral(eval(f'pt_start{i+1}'), eval(f'pt_end{i+1}'), Vm_matrix[i], stroke1, channel, modal_Ex_lossy[i, :, :], dx, 0)
    modal_Ex_source2[i, :] = integral(eval(f'pt_start{i+1}'), eval(f'pt_end{i+1}'), Vm_matrix[i], stroke1, channel, modal_Ex_lossy[i, :, :], dx, 1)

Is_inc1 = -(Ti @ Zm_matrix_inv @ modal_Ex_source1)  # 加了负号，保证和文献中电流源的方向一致
Is_inc2 = -(-Ti @ Zm_matrix_inv @ modal_Ex_source2) # 加了负号，保证和文献中电流源的方向一致

tau_m = np.abs(pt_end[-1, 0] - pt_start[0, 0]) / Vm_matrix
Y_matrix_block1 = Ti @ Zm_matrix_inv @ Tv_inv
Y_matrix_block2 = Ti @ Zm_matrix_inv @ Tv_inv
Y_matrix = np.block([[Y_matrix_block1, np.zeros((Y_matrix_block1.shape[0], Y_matrix_block2.shape[1]))],
                      [np.zeros((Y_matrix_block2.shape[0], Y_matrix_block1.shape[1])), Y_matrix_block2]])
Z_matrix = np.sqrt(L_matrix / C_matrix)

def calculate_resistances_from_admittance(Y):
    n = Y.shape[0]
    
    # 初始化矩阵
    R_between = np.full((n, n), np.inf)
    R_to_ground = np.full(n, np.inf)
    
    # 计算节点之间的导纳 G_between
    G_between = -Y.copy()
    np.fill_diagonal(G_between, 0)
    
    # 计算节点之间的电阻 R_between
    with np.errstate(divide='ignore', invalid='ignore'):
        R_between = np.where(G_between != 0, 1 / G_between, np.inf)
    
    # 计算各节点接地的总导纳 G_to_ground
    G_to_ground = np.diag(Y) - np.sum(G_between, axis=1)
    
    # 计算各节点对地的电阻 R_to_ground
    # with np.errstate(divide='ignore', invalid='ignore'):
    #     R_to_ground = np.where(G_to_ground != 0, 1 / G_to_ground, np.inf)
    np.fill_diagonal(R_between, 1 / G_to_ground)
    R = R_between
    return R
R_matrix = calculate_resistances_from_admittance(Y_matrix_block1)
# df1 = pd.DataFrame(Is_inc1)
# df2 = pd.DataFrame(Is_inc2)
# df1.to_excel('D:/Documents/ATP_related/Projects/DEPACT/Is_inc1.xlsx', index=False, header=False)
# df2.to_excel('D:/Documents/ATP_related/Projects/DEPACT/Is_inc2.xlsx', index=False, header=False)
# write_to_txt(Is_inc1[0, :], f'D:/Documents/ATP_related/Projects/DEPACT/Is_inc1_A_seg{seg_seq}.txt')
# write_to_txt(Is_inc1[1, :], f'D:/Documents/ATP_related/Projects/DEPACT/Is_inc1_B_seg{seg_seq}.txt')
# write_to_txt(Is_inc1[2, :], f'D:/Documents/ATP_related/Projects/DEPACT/Is_inc1_C_seg{seg_seq}.txt')
# write_to_txt(Is_inc2[0, :], f'D:/Documents/ATP_related/Projects/DEPACT/Is_inc2_A_seg{seg_seq}.txt')
# write_to_txt(Is_inc2[1, :], f'D:/Documents/ATP_related/Projects/DEPACT/Is_inc2_B_seg{seg_seq}.txt')
# write_to_txt(Is_inc2[2, :], f'D:/Documents/ATP_related/Projects/DEPACT/Is_inc2_C_seg{seg_seq}.txt')

def formula1(i, j, omega, h, constants, ground):
    gamma = np.sqrt(1j * omega * constants.mu0 * (ground.sig + 1j * ground.epr * constants.ep0))
    return 1j * omega * constants.mu0 / (2 * np.pi) * np.log((1 + gamma * h[i]) / (gamma * h[j]))

def formula2(i, j, omega, h, r, constants, ground):
    gamma = np.sqrt(1j * omega * constants.mu0 * (ground.sig + 1j * ground.epr * constants.ep0))
    tmp1 = (1 + gamma * (h[i] +h[j] / 2))**2 + (gamma * (r[i, j] / 2))**2
    tmp2 = (gamma * (h[i] + h[j] / 2))**2 + (gamma * (r[i, j] / 2))**2
    return 1j * omega * constants.mu0 / (4 * np.pi) * np.log(tmp1 / tmp2)
    
r = np.abs(height[:, np.newaxis] - height)  # 导线间距矩阵
omega_sample = np.array(
        [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000,
         10000000]).reshape(1, -1)
# 初始化阻抗矩阵Z
Z = np.zeros((height.size, height.size, omega_sample.size), dtype=complex)
for i in range(height.size):
    for j in range(height.size):
        if i == j:
            for k in range(omega_sample.size):
                Z[i, j, k] = formula1(i, j, omega_sample[0, k], height, constants, ground)
        else:
            for k in range(omega_sample.size):
                Z[i, j, k] = formula2(i, j, omega_sample[0, k], height, r, constants, ground)
N_fit = 9
R0, L0, Rn, Ln, Z_fit = vecfit_kernel_Z_Ding(Z, omega_sample / 2 / np.pi, 9, vf_mod='D')
# plt.plot(stroke1.t_us, modal_Ex_source1[0, :])
# plt.show()
plt.plot(stroke1.t_us, Is_inc1[0, :])
plt.plot(stroke1.t_us, Is_inc1[1, :])
plt.plot(stroke1.t_us, Is_inc1[2, :])
plt.show()
print('1')