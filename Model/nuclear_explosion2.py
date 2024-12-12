import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from Contant import Constant
from Ground import Ground
from Vector_Fitting.Calculators.vecfit_kernel_z import vecfit_kernel_Z_Ding
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve
from scipy.io import savemat


constant = Constant()
ground = Ground(sig=0.01, mur=1, epr=10)

a = np.pi / 6
theta_i = np.pi / 4
# Nt = 1000+1
t = np.linspace(0, 3e-7, 10000+1)
dt = t[1] - t[0]
alpha = 4e7
beta = 6e8
k = 1.3
amplitude = 50e3
waveform = k * amplitude * (np.exp(-alpha * t) - np.exp(-beta * t))
# waveform = k * amplitude * np.cos(1e8 * t)
omega = 2 * np.pi * 1e8
Nd = 16
position = np.array([2, 0, -1])  # 观察点坐标

def delay_waveform(wave, t, t0):
    """
    将波形整体时延 t0，小于 t0 的时间设置为零。

    参数：
    - wave: 原始波形序列，numpy 数组
    - t: 时间数组，与 wave 长度相同
    - t0: 时延，浮点数

    返回：
    - new_wave: 延迟后的新波形，numpy 数组
    """
    # 确保 wave 和 t 是 numpy 数组
    wave = np.array(wave)
    t = np.array(t)

    # 创建新波形，初始化为零
    new_wave = np.zeros_like(wave)

    # 计算延迟后的时间数组
    t_delayed = t - t0

    # 找到 t_delayed >= 0 的索引
    valid_indices = t_delayed >= 0

    # 对于有效的时间点，进行插值
    new_wave[valid_indices] = np.interp(t_delayed[valid_indices], t, wave)

    return new_wave
def cal_waveform(t, t0):
    """
    :param t: 时刻点序列
    :param t0: 时间延迟
    :return:
    """
    waveform = np.where(t > t0, k * amplitude * (np.exp(-alpha * (t - t0)) - np.exp(-beta * (t - t0))), 0)
    # waveform = k * amplitude * np.cos(1e8 * (t - t0))
    return waveform

def calculate_time_domain_response(waveform, H_function, t, dt, omega_array, Nd):
    """
    首先对waveform进行线性插值，然后计算waveform和传递函数H_function的卷积，最后还原至原时间尺度
    :param waveform:
    :param H_function: 要求是三维的，1*1*n
    :param t:
    :param dt: 时间步长
    :param omega_array: 选取的频率点，shape为1*n
    :return:
    """
    R0, L0, Rn, Ln, Zfit = vecfit_kernel_Z_Ding(H_function, omega_array / (2 * np.pi), Nfit=Nd)
    a00, Nt = waveform.shape
    t_ob = t[-1]  # 总观测时间
    conv_2 = 2  # 上采样倍数
    dt0 = dt / conv_2  # 上采样后的新时间步长

    # 原始时间点和数据
    x = np.linspace(0, t_ob, Nt)  # 原始时间点
    y = waveform  # 原始磁场数据

    # 上采样后的新时间点
    xi = np.linspace(0, t_ob, 2 * Nt)  # 上采样的时间点

    # 对 HR0 进行插值以提高时间分辨率
    interp_func = interp1d(x, y, kind='cubic', axis=1, fill_value="extrapolate")
    waveform_i_interp = interp_func(xi)  # 上采样后的电场数据，形状为 (空间点数, 2 * Nt)

    # 如果只有一个空间点，转置数组以确保时间维度在轴 0 上
    if a00 == 1:
        waveform_i_interp = waveform_i_interp.T  # 形状变为 (2 * Nt, 1)

    # 确保时间维度在轴 0 上
    if waveform_i_interp.shape[0] != 2 * Nt:
        waveform_i_interp = waveform_i_interp.T  # 现在 waveform_i_interp 的形状为 (2 * Nt, 空间点数)
    Ntt = waveform_i_interp.shape[0]  # 上采样后的时间点数

    # 计算 waveform_i_interp 的数值导数
    waveform_i_all_diff = np.zeros_like(waveform_i_interp)
    waveform_i_all_diff[0, :] = waveform_i_interp[0, :] / dt0
    waveform_i_all_diff[1:, :] = np.diff(waveform_i_interp, axis=0) / dt0

    # 计算 R0 和 L0 组件的贡献
    ee0 = R0 * waveform_i_interp  # 形状为 (Ntt, 空间点数)
    eeL = L0 * waveform_i_all_diff  # 形状为 (Ntt, 空间点数)

    # 为 Rn 和 Ln 组件的卷积做准备
    # 时间索引
    tt = np.arange(1, Ntt + 1).reshape(Ntt, 1)  # 形状为 (Ntt, 1)
    # Rn 和 Ln 数组
    Rn2 = Rn[0, :Nd]  # 形状为 (Nd,)`
    Ln2 = Ln[0, :Nd]  # 形状为 (Nd,)

    # 计算用于卷积核的指数项
    exponent = - (Rn2 / Ln2) * tt * dt0  # 形状为 (Ntt, Nd)
    ee = - (Rn2 ** 2 / Ln2) * np.exp(exponent)  # 形状为 (Ntt, Nd)

    # 初始化卷积结果的累加数组
    ee_conv_sum = np.zeros_like(waveform_i_interp)  # 形状为 (Ntt, 空间点数)

    # 对每个空间点执行卷积
    for ii in range(waveform_i_interp.shape[1]):
        H_ii = waveform_i_interp[:, ii]  # 第 ii 个空间点的磁场，形状为 (Ntt,)
        conv_total = np.zeros(Ntt + Ntt - 1)
        # 对每个极点进行卷积
        for jj in range(Nd):
            ee_jj = ee[:, jj]  # 第 jj 个极点的卷积核，形状为 (Ntt,)
            conv_result = dt0 * fftconvolve(H_ii, ee_jj, mode='full')  # 卷积结果，形状为 (2*Ntt - 1,)
            conv_total += conv_result  # 累加卷积结果
        # 存储卷积结果（截断到原始长度）
        ee_conv_sum[:, ii] = conv_total[:Ntt]

    # 汇总所有贡献，并下采样回原始时间步长
    ee_all = ee0[::conv_2, :] + eeL[::conv_2, :] + ee_conv_sum[::conv_2, :]  # 形状为 (Nt, 空间点数)
    return np.squeeze(ee_all)


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
    unit_E_incide_TM_direction = np.array([np.cos(theta_i), 0, np.sin(theta_i)])
    unit_E_incide_direction = np.array([np.cos(a) * np.cos(theta_i), -np.sin(a), np.cos(a) * np.sin(theta_i)])

    unit_E_reflect_TE_direction = np.array([0, -1, 0])
    unit_E_reflect_TM_direction = np.array([-np.cos(theta_i), 0, np.sin(theta_i)])
    unit_E_reflect_direction = np.array([-np.cos(a) * np.cos(theta_i), -np.sin(a), np.cos(a) * np.sin(theta_i)])

    # sample_omega = np.array(
    #     [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000,
    #      10000000, 5e7, 1e8, 5e8, 1e9, 5e9, 1e10, 5e10]).reshape(1, -1)  # 用于拟合的频率点
    sample_omega = np.logspace(-4, 12, num=50+1, base=10).reshape(1, -1) * 2 * np.pi
    # sample_omega = np.linspace(1e-12, 1e12, 50+1).reshape(1, -1)


    tmp = ground.epr * (1 + ground.sig / (1j * sample_omega * ground.epr * constant.ep0))
    H_function_r_TE = (-np.cos(theta_i) + np.sqrt(ground.epr * (1 + ground.sig / (1j * sample_omega * ground.epr * constant.ep0)) - np.sin(theta_i) ** 2)) / \
                       (np.cos(theta_i) + np.sqrt(ground.epr * (1 + ground.sig / (1j * sample_omega * ground.epr * constant.ep0)) - np.sin(theta_i) ** 2))
    H_function_r_TM = (-tmp * np.cos(theta_i) + np.sqrt(tmp - np.sin(theta_i)**2)) / (tmp * np.cos(theta_i) + np.sqrt(tmp - np.sin(theta_i)**2))
    H_function_r_TE = H_function_r_TE.reshape((1, 1, sample_omega.shape[1]))
    H_function_r_TM = H_function_r_TM.reshape((1, 1, sample_omega.shape[1]))

    # R0_r_TE, L0_r_TE, Rn_r_TE, Ln_r_TE, Zfit_r_TE = vecfit_kernel_Z_Ding(H_function_r_TE, sample_omega / (2 * np.pi), Nfit=Nd)
    # R0_r_TM, L0_r_TM, Rn_r_TM, Ln_r_TM, Zfit_r_TM = vecfit_kernel_Z_Ding(H_function_r_TM, sample_omega / (2 * np.pi), Nfit=Nd)

    waveform_r_TE = calculate_time_domain_response(waveform.reshape((1, -1)) * np.sin(a), H_function_r_TE, t, dt, sample_omega, Nd)
    waveform_r_TM = calculate_time_domain_response(waveform.reshape((1, -1)) * np.cos(a), H_function_r_TM, t, dt, sample_omega, Nd)

    # 加上反射波延迟
    waveform_r_TE = delay_waveform(waveform_r_TE, t, t0_r)
    waveform_r_TM = delay_waveform(waveform_r_TM, t, t0_r)
    waveform_r_TE_amp = waveform_r_TE.reshape((-1, 1)) * unit_E_reflect_TE_direction @ unit_E_reflect_TE_direction
    waveform_r_TM_amp = waveform_r_TM.reshape((-1, 1)) * unit_E_reflect_TM_direction @ unit_E_reflect_TM_direction
    waveform_r_vector = waveform_r_TE.reshape((-1, 1)) * unit_E_reflect_TE_direction + waveform_r_TM.reshape((-1, 1)) * unit_E_reflect_TM_direction
    # waveform_r_amp = waveform_r_vector @ unit_E_reflect_direction
    waveform_r_down_x_amp = waveform_r_vector * np.array([1, 0, 0]) @ np.array([1, 0, 0])
    waveform_r_down_y_amp = waveform_r_vector * np.array([0, 1, 0]) @ np.array([0, 1, 0])
    waveform_r_down_z_amp = waveform_r_vector * np.array([0, 0, 1]) @ np.array([0, 0, 1])

    # 观察点处的入射波
    waveform_i = delay_waveform(waveform, t, t0_i)
    waveform_i_TE_amp = waveform_i.reshape((-1, 1)) * np.sin(a) * unit_E_incide_TE_direction @ unit_E_incide_TE_direction
    waveform_i_TM_amp = waveform_i.reshape((-1, 1)) * np.cos(a) * unit_E_incide_TM_direction @ unit_E_incide_TM_direction
    waveform_i_vector = waveform_i.reshape((-1, 1)) * unit_E_incide_direction
    waveform_i_down_x_amp = waveform_i_vector * np.array([1, 0, 0]) @ np.array([1, 0, 0])
    waveform_i_down_y_amp = waveform_i_vector * np.array([0, 1, 0]) @ np.array([0, 1, 0])
    waveform_i_down_z_amp = waveform_i_vector * np.array([0, 0, 1]) @ np.array([0, 0, 1])

    # waveform_total_TM_vector = waveform_i.reshape((-1, 1)) * np.cos(a) * unit_E_incide_TM_direction + waveform_r_TM.reshape((-1, 1)) * unit_E_reflect_TM_direction
    # waveform_total_TM_direction = waveform_total_TM_vector / np.linalg.norm(waveform_total_TM_vector, axis=1).reshape((-1, 1))
    waveform_total_vector = waveform_i_vector + waveform_r_vector
    waveform_total_down_x = waveform_total_vector * np.array([1, 0, 0]) @ np.array([1, 0, 0])
    waveform_total_down_y = waveform_total_vector * np.array([0, 1, 0]) @ np.array([0, 1, 0])
    waveform_total_down_z = waveform_total_vector * np.array([0, 0, 1]) @ np.array([0, 0, 1])
    waveform_total_amp = np.linalg.norm(waveform_total_vector, axis=1)

    # 原始波形
    waveform_TE_amp = waveform.reshape((-1, 1)) * np.sin(a) *  unit_E_incide_TE_direction @ unit_E_incide_TE_direction
    waveform_TM_down_x = waveform.reshape((-1, 1)) * np.cos(a) * unit_E_incide_TM_direction * np.array([1, 0, 0]) @ np.array([1, 0, 0])
    waveform_TM_down_z = waveform.reshape((-1, 1)) * np.cos(a) * unit_E_incide_TM_direction * np.array([0, 0, 1]) @ np.array([0, 0, 1])
    waveform_TM_amp = waveform.reshape((-1, 1)) * np.cos(a) * unit_E_incide_TM_direction @ unit_E_incide_TM_direction

    # 画图
    # plt.plot(t, waveform, label='original')
    # plt.plot(t, waveform_i, label='incide')
    # plt.plot(t, waveform_r_amp, label='reflect')
    # plt.plot(t, waveform_total_amp, label='total')
    # plt.legend()
    # 创建一个图形窗口，包含1行2列的两个子图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 5))  # figsize 控制整个图形窗口的大小
    # 在第一个子图上绘图
    ax1.plot(t, waveform_TE_amp, 'red', label='original_label')  # 'r-' 表示红色实线
    ax1.plot(t, waveform_i_TE_amp, 'blue', label='incide')  # 'r-' 表示红色实线
    ax1.plot(t, waveform_r_TE_amp, 'orange', label='reflect')  # 'r-' 表示红色实线
    # ax1.plot(t, waveform_total_down_y, 'purple', label='total')  # 'r-' 表示红色实线
    ax1.set_title('TE(y_axis)')  # 设置第一个子图的标题
    ax1.set_xlabel('t / s')  # 设置x轴标签
    ax1.set_ylabel('E / V')  # 设置y轴标签

    # 在第二个子图上绘图
    ax2.plot(t, waveform_TM_amp, 'red', label='original_label')  # 'r-' 表示红色实线
    ax2.plot(t, waveform_i_TM_amp, 'blue', label='incide')  # 'r-' 表示红色实线
    ax2.plot(t, waveform_r_TM_amp, 'orange', label='reflect')  # 'r-' 表示红色实线
    ax2.set_title('TM')  # 设置第一个子图的标题
    ax2.set_xlabel('t / s')  # 设置x轴标签
    ax2.set_ylabel('E / V')  # 设置y轴标签

    # 在第三个子图上绘图
    ax3.plot(t, waveform_TM_down_x, 'red', label='original_label')  # 'r-' 表示红色实线
    ax3.plot(t, waveform_i_down_x_amp, 'blue', label='incide')  # 'r-' 表示红色实线
    ax3.plot(t, waveform_r_down_x_amp, 'orange', label='reflect')  # 'r-' 表示红色实线
    # ax3.plot(t, waveform_total_down_x, 'purple', label='total')  # 'r-' 表示红色实线
    ax3.set_title('x_axis')  # 设置第一个子图的标题
    ax3.set_xlabel('t / s')  # 设置x轴标签
    ax3.set_ylabel('E / V')  # 设置y轴标签

    # 在第四个子图上绘图
    ax4.plot(t, waveform_TM_down_z, 'red', label='original_label')  # 'r-' 表示红色实线
    ax4.plot(t, waveform_i_down_z_amp, 'blue', label='incide')  # 'r-' 表示红色实线
    ax4.plot(t, waveform_r_down_z_amp, 'orange', label='reflect')  # 'r-' 表示红色实线
    # ax4.plot(t, waveform_total_down_z, 'purple', label='total')  # 'r-' 表示红色实线
    ax4.set_title('z_axis')  # 设置第一个子图的标题
    ax4.set_xlabel('t / s')  # 设置x轴标签
    ax4.set_ylabel('E / V')  # 设置y轴标签

    # 调整子图间距
    plt.subplots_adjust(wspace=0.3)  # wspace 控制子图之间的宽度空间
    plt.legend()
    plt.show()

else:
    trans_position = np.array([0, 0, 0])
    trans_position[0] = position[0] - (-position[2]) * np.tan(theta_i)  # x0 - (-z0) * tan(θ)，z0是负数
    ki = omega / constant.vc
    kt_x = ki * np.sin(theta_i)
    # sample_omega = np.array(
    #     [1e-10, 5e-10, 1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000,
    #      10000000, 5e7, 1e8, 5e8, 1e9, 5e9, 1e10, 5e10]).reshape(1, -1)  # 用于拟合的频率点
    # sample_omega = np.linspace(1e-12, 1e12, 25).reshape(1, -1)
    sample_low_freq_omega = np.logspace(-4, 2, num=12+1, base=10).reshape(1, -1) * 2 * np.pi
    sample_high_freq_omega = np.logspace(7, 10, num=10+1, base=10).reshape(1, -1) * 2 * np.pi
    sample_middle_freq_omega = np.logspace(2, 7, num=10+1, base=10).reshape(1, -1) * 2 * np.pi
    sample_omega = np.hstack((sample_low_freq_omega, sample_middle_freq_omega, sample_high_freq_omega))

    sin_theta_t = np.sqrt(1 / (ground.epr * (1 + ground.sig / (1j * sample_omega * ground.epr * constant.ep0)))) * np.sin(theta_i)
    theta_t = np.arcsin(sin_theta_t)  # 实际的折射角
    kt_z = - np.sqrt(sample_omega**2 * ground.mur * constant.mu0 * (ground.epr * constant.ep0 + ground.sig / (1j * sample_omega)) -
                     sample_omega**2 * 1 * constant.mu0 * 1 * constant.ep0 * np.sin(theta_i)**2)

    unit_ki_direction = np.array([np.sin(theta_i),
                         0,
                         -np.cos(theta_i)])
    unit_kt_x_direction = np.array([1, 0, 0])   # 沿x方向的分量
    ki_vector = ki * unit_ki_direction
    kt_x_vector = kt_x * unit_kt_x_direction
    t0_i = ki_vector @ position / omega
    t0_t = (ki_vector * np.sin(theta_i) @ np.array([position[0], 0, 0])) / omega

    # 入射和折射TE与TM波的单位方向向量
    unit_E_incide_TE_direction = np.array([0, -1, 0])
    unit_E_incide_TM_direction = np.array([np.cos(theta_i), 0, np.sin(theta_i)])
    unit_E_incide_direction = np.array([np.cos(a) * np.cos(theta_i), -np.sin(a), np.cos(a) * np.sin(theta_i)])

    # unit_E_transmit_TE_direction = np.array([0, -1, 0])
    # unit_E_transmit_TM_direction = np.array([np.cos(theta_t), 0, np.sin(theta_t)])
    # unit_E_transmit_direction = np.array([np.cos(a) * np.cos(theta_t), -np.sin(a), np.cos(a) * np.sin(theta_t)])

    tmp = ground.epr * (1 + ground.sig / (1j * sample_omega * ground.epr * constant.ep0))

    t_TE_coefficient = 2 * np.cos(theta_i) / (np.cos(theta_i) + np.sqrt(tmp - np.sin(theta_i)**2))
    t_TM_coefficient = (2 * np.sqrt(tmp) * np.cos(theta_i)) / (tmp * np.cos(theta_i) + np.sqrt(tmp - np.sin(theta_i)**2))

    H_function_t_x = t_TE_coefficient * np.exp(- 1j * kt_z * position[2]) * np.cos(theta_t) * np.exp(1j * (-sample_omega / constant.vc) * np.cos(theta_i))
    H_function_t_y = t_TM_coefficient * np.exp(- 1j * kt_z * position[2]) * np.exp(1j * (-sample_omega / constant.vc) * np.cos(theta_i))
    # H_function_t_y = t_TE_coefficient * np.exp(- 1j * kt_z * position[2]) * np.cos(theta_t)
    H_function_t_z = t_TE_coefficient * np.exp(- 1j * kt_z * position[2]) * np.sin(theta_t) * np.exp(1j * (-sample_omega / constant.vc) * np.cos(theta_i))
    H_function_t_x = H_function_t_x.reshape((1, 1, sample_omega.shape[1]))
    H_function_t_y = H_function_t_y.reshape((1, 1, sample_omega.shape[1]))
    H_function_t_z = H_function_t_z.reshape((1, 1, sample_omega.shape[1]))
    # R0_r_TE, L0_r_TE, Rn_r_TE, Ln_r_TE, Zfit_r_TE = vecfit_kernel_Z_Ding(H_function_r_TE, sample_omega / (2 * np.pi), Nfit=Nd)
    # R0_r_TM, L0_r_TM, Rn_r_TM, Ln_r_TM, Zfit_r_TM = vecfit_kernel_Z_Ding(H_function_r_TM, sample_omega / (2 * np.pi), Nfit=Nd)

    waveform_t_x = calculate_time_domain_response(waveform.reshape((1, -1)) * np.cos(a), H_function_t_x, t, dt, sample_omega, Nd)
    waveform_t_y = calculate_time_domain_response(waveform.reshape((1, -1)) * (-np.sin(a)), H_function_t_y, t, dt, sample_omega, Nd)
    waveform_t_z = calculate_time_domain_response(waveform.reshape((1, -1)) * np.cos(a), H_function_t_x, t, dt, sample_omega, Nd)

    # 加上透射波水平延迟
    waveform_t_x = delay_waveform(waveform_t_x, t, t0_t)
    waveform_t_y = delay_waveform(waveform_t_y, t, t0_t)
    waveform_t_z = delay_waveform(waveform_t_z, t, t0_t)
    waveform_t_vector = np.vstack((waveform_t_x, waveform_t_y, waveform_t_z)).T
    # waveform_t_amp = np.linalg.norm(waveform_t_vector, axis=1)
    waveform_t_amp = np.sqrt(waveform_t_x**2 + waveform_t_y**2 + waveform_t_z**2)
    # waveform_t_amp = waveform_t_vector @ np.array([0.6087, 0.5089, 0.6085])
    # waveform_t_direction = waveform_t_vector / waveform_t_amp

    waveform_i = delay_waveform(waveform, t, t0_i)
    waveform_i_vector = waveform_i.reshape((-1, 1)) * unit_E_incide_direction

    plt.plot(t, waveform, label='original')
    # plt.plot(t, waveform_i, label='incide')
    plt.plot(t, waveform_t_amp, label='transmit')
    plt.legend()

    plt.show()

#





