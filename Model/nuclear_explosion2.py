import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from Contant import Constant
from Ground import Ground
from Vector_Fitting.Calculators.vecfit_kernel_z import vecfit_kernel_Z_Ding
from matplotlib.animation import FuncAnimation

constant = Constant()
ground = Ground(sig=0.01, mur=1, epr=10)

a = np.pi / 6
theta_i = np.pi / 4
t = np.linspace(0, 1e-7, 1000)
alpha = 4e7
beta = 6e8
k = 1.3
amplitude = 50e3
waveform = k * amplitude * (np.exp(-alpha * t) - np.exp(-beta * t))
omega = 2 * np.pi * 1e8

def cal_waveform(t, t0):
    """
    :param t: 时刻点序列
    :param t0: 时间延迟
    :return:
    """
    waveform = np.where(t > t0, k * amplitude * (np.exp(-alpha * (t - t0)) - np.exp(-beta * (t - t0))), 0)
    return waveform


waveform = cal_waveform(t, 0)
position = np.array([2, 0, 1])  # 观察点坐标
if position[2] > 0:
    trans_position = np.array([0, 0, 0])
    trans_position[0] = position[0] - position[2] * np.tan(theta_i)
    ki = omega / constant.vc
    kr = ki
    unit_ki_direction = np.array([np.sin(theta_i),
                         0,
                         -np.cos(theta_i)])
    unit_kr_direction = np.array([np.sin(theta_i),
                         0,
                         np.cos(theta_i)])
    ki_vector = ki * unit_ki_direction
    kr_vector = kr * unit_kr_direction
    t0_i = ki_vector @ position / omega
    t0_r = (ki_vector @ trans_position + kr_vector @ (position - trans_position)) / omega

    # 入射和反射TE与TM波的单位方向向量
    unit_E_incide_TE_direction = np.array([0, -1, 0])
    unit_E_inciede_TM_direction = np.array([np.cos(theta_i), 0, np.sin(theta_i)])
    unit_E_incide_direction = np.array([np.cos(a) * np.cos(theta_i), -np.sin(a), np.cos(a) * np.sin(theta_i)])

    unit_E_reflect_TE_direction = np.array([0, -1, 0])
    unit_E_reflect_TM_direction = np.array([-np.cos(theta_i), 0, np.sin(theta_i)])
    unit_E_reflect_direction = np.array([-np.cos(a) * np.cos(theta_i), -np.sin(a), np.cos(a) * np.sin(theta_i)])

    sample_omega = np.array(
        [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000,
         10000000]).reshape(1, -1)  # 用于拟合的频率点

    H_function_r_TE = (np.cos(theta_i) - np.sqrt(ground.epr * (1 + ground.sig / (1j * sample_omega * ground.epr * constant.ep0)) - np.sin(theta_i) ** 2)) / \
                       (np.cos(theta_i) + np.sqrt(ground.epr * (1 + ground.sig / (1j * sample_omega * ground.epr * constant.ep0)) - np.sin(theta_i) ** 2))
    H_function_r_TM = ((2 * np.cos(theta_i)) /
                       (np.cos(theta_i) + np.sqrt(ground.epr * (1 + ground.sig / (1j * sample_omega * ground.epr * constant.ep0)) - np.sin(theta_i) ** 2)))
    H_function_r_TE = H_function_r_TE.reshape((1, 1, 19))
    H_function_r_TM = H_function_r_TM.reshape((1, 1, 19))
    R0_r_TE, L0_r_TE, Rn_r_TE, Ln_r_TE, Zfit_r_TE = vecfit_kernel_Z_Ding(H_function_r_TE, sample_omega / (2 * np.pi), Nfit=9)
    R0_r_TM, L0_r_TM, Rn_r_TM, Ln_r_TM, Zfit_r_TM = vecfit_kernel_Z_Ding(H_function_r_TE, sample_omega / (2 * np.pi), Nfit=9)

    #
    waveform_i = cal_waveform(t, t0_i)


plt.plot(t, waveform)
plt.plot(t, waveform_i)
plt.show()




