import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
from Contant import Constant
from Ground import Ground
from matplotlib.animation import FuncAnimation


class NuclearExplosionSource:
    def __init__(self, f, k, amplitude, alpha, beta, phi0, theta_i, ground: Ground):
        """
        初始化核爆源模型。

        参数：
        - t: 时间
        - f: 频率
        - k: 脉冲幅值校正系数
        - amplitude：波形的峰值幅度
        - alpha：第一个指数项的衰减常数
        - beta：第二个指数项的衰减常数
        - theta_i：入射角，单位为°
        """
        self.omega = 2 * np.pi * f
        self.k = k
        self.amplitude = amplitude
        self.alpha = alpha
        self.beta = beta
        self.phi0 = phi0
        self.theta_i = np.radians(theta_i)
        self.ground = ground

        self.vc = Constant().vc
        self.k0 = 2 * np.pi * 1e8 / self.vc # 波矢幅值
        # 根据入射角计算方向向量（假设入射面是x-z平面）
        self.direction = np.array([
            np.sin(self.theta_i),
            0,
            - np.cos(self.theta_i)
        ])
        self.k_vector = self.k0 * self.direction

        # self.waveform = self.k * self.amplitude * np.cos(self.omega * self.t)

    def propagate(self, a, t, position):
        """
        根据平面波公式计算某位置某时刻的波形
        a: 电场矢量与垂直极化波的夹角
        """
        t = np.array(t)
        # 给观察点的坐标赋值
        x0 = position[0]
        y0 = position[1]
        z0 = position[2]

        # 计算折射系数与本征阻抗
        n1 = 1
        eta1 = 1  # 空气本征阻抗
        n2 = np.sqrt(self.ground.mur * self.ground.epr)
        eta2 = np.sqrt(self.ground.mur / self.ground.epr)  # 大地本征阻抗
        # 折射角计算
        sin_theta_t = (n1 / n2) * np.sin(self.theta_i)
        cos_theta_t = np.sqrt(1 - sin_theta_t ** 2)
        tan_theta_t = sin_theta_t / cos_theta_t

        '''入射相位差，TE和TM波'''
        incide_phase = np.dot(self.k_vector, position)
        E0_incide = self.k * self.amplitude
        E_incide_waveform = E0_incide * np.cos(self.omega * t - incide_phase + self.phi0)
        # TE波分量
        E_incide_TE_direction = np.array([0, -1, 0])
        E_incide_TE = np.array(
            [E0_incide * np.sin(a) * np.cos(self.omega * dt - incide_phase + self.phi0) * E_incide_TE_direction for
             dt in t])

        # TM波分量
        E_incide_TM_direction = np.array([np.cos(self.theta_i), 0, np.sin(self.theta_i)])
        E_incide_TM = np.array(
            [E0_incide * np.cos(a) * np.cos(self.omega * dt - incide_phase + self.phi0) * E_incide_TM_direction for
             dt in t])

        # 考虑反射波与透射波
        if position[2] >= 0:
            '''反射相位差，TE和TM波'''
            # 反射波矢
            reflect_direction = self.direction
            reflect_direction[2] = - self.direction[2]
            kr_vector = self.k0 * reflect_direction
            # 反射点坐标
            reflect_position = np.array([x0-abs(z0 * np.tan(self.theta_i)), 0, 0])
            # 相位差，时延
            reflect_phase = np.dot(self.k_vector, reflect_position) + np.dot(kr_vector, position-reflect_position)
            # reflect_waveform = self.k * self.amplitude * np.cos(self.omega * t - reflect_phase + self.phi0)
            t0 = reflect_phase / self.omega

            # 计算TE波的反射系数
            r_perpendicular = (eta2 * np.cos(self.theta_i) - eta1 * cos_theta_t) / \
                              (eta2 * np.cos(self.theta_i) + eta1 * cos_theta_t)
            self.r_perpendicular = r_perpendicular
            # 计算TM波的反射系数
            r_parallel = (eta2 * cos_theta_t - eta1 * np.cos(self.theta_i)) / \
                         (eta2 * cos_theta_t + eta1 * np.cos(self.theta_i))
            self.r_parallel = r_parallel

            # 反射波的TE波分量
            E0_reflect_TE = self.k * self.amplitude * np.sin(a) * r_perpendicular  # 反射波的垂直极化波幅值
            E_reflect_TE_direction = np.array([0, -1, 0])
            E_reflect_TE = np.array(
                [E0_reflect_TE * np.cos(self.omega * dt - reflect_phase + self.phi0) * E_reflect_TE_direction for dt in
                 t])

            # 反射波的TM波分量
            E0_reflect_TM = self.k * self.amplitude * np.cos(a) * r_parallel  # 反射波的平行极化波幅值
            E_reflect_TM_direction = np.array([- np.cos(self.theta_i), 0, np.sin(self.theta_i)])  # 反射波的平行极化波方向
            E_reflect_TM = np.array(
                [E0_reflect_TM * np.cos(self.omega * dt - reflect_phase + self.phi0) * E_reflect_TM_direction for dt in
                 t])  # 反射波的而平行极化波矢量

            # 反射波矢量
            reflect_waveform = E_reflect_TE + E_reflect_TM
            # 入射波与反射波合成矢量
            total_waveform = E_incide_TE + E_incide_TM + E_reflect_TE + E_reflect_TM
            return t0, np.squeeze(E_incide_waveform), np.squeeze(reflect_waveform), np.squeeze(total_waveform)

        elif position[2] < 0:
            '''折射相位差，假如观察点在地下，则z坐标为负'''
            # 计算折射波矢
            kt0 = self.k0 * n2  # 折射波矢幅值
            transmit_direction = np.array([
                sin_theta_t,
                0,
                -cos_theta_t
            ])  # 折射波方向向量
            kt_vector = kt0 * transmit_direction
            # 折射点坐标
            transmit_position = np.array([x0-abs(z0 * tan_theta_t), 0, 0])
            # 相位差，时延
            transmit_phase = np.dot(self.k_vector, transmit_position) + np.dot(kt_vector, position - transmit_position)
            transmit_waveform = self.k * self.amplitude * np.cos(self.omega * t - transmit_phase + self.phi0)
            t0 = transmit_phase / self.omega

            # 计算TE波透射系数
            t_perpendicular = (2 * eta2 * np.cos(self.theta_i)) / \
                              (eta2 * np.cos(self.theta_i) + eta1 * cos_theta_t)
            self.t_perpendicular = t_perpendicular

            # 计算TM波透射系数
            t_parallel = (2 * eta2 * np.cos(self.theta_i)) / \
                         (eta2 * cos_theta_t + eta1 * np.cos(self.theta_i))
            self.t_parallel = t_parallel

            # 透射波
            E0_transmit_TE = self.k * self.amplitude * np.sin(a) * t_perpendicular  # 折射波的垂直极化波幅值
            E_transmit_TE_direction = np.array([0, -1, 0])
            E_transmit_TE =  np.array(
                [E0_transmit_TE * np.cos(self.omega * dt - transmit_phase + self.phi0) * E_transmit_TE_direction for dt in
                 t])

            E0_transmit_TM = self.k * self.amplitude * np.cos(a) * t_parallel  # 折射波的平行极化波幅值
            E_transmit_TM_direction = np.array([cos_theta_t, 0, sin_theta_t])  # 折射波的而平行极化波方向
            E_transmit_TM = np.array(
                [E0_transmit_TM * np.cos(self.omega * dt - transmit_phase + self.phi0) * E_transmit_TM_direction for dt in
                 t])  # 折射波的而平行极化波矢量

            # 总场=折射
            total_waveform = E_transmit_TE + E_transmit_TM
            return t0, np.squeeze(E_incide_waveform), np.squeeze(total_waveform)


    def transform_coordinates(self, positions, phi):
        """
        将坐标系旋转，使得导线与 x 轴平行。

        参数：
        - positions：位置数组（N x 3）
        - phi：导线与入射波的夹角（以度为单位）

        返回：
        - transformed_positions：旋转后的坐标数组
        """
        phi_rad = np.radians(phi)
        # 绕 z 轴旋转矩阵（角度为 -phi）
        R = np.array([
            [np.cos(phi_rad), np.sin(phi_rad), 0],
            [-np.sin(phi_rad), np.cos(phi_rad), 0],
            [0, 0, 1]
        ])
        # 对所有位置进行旋转
        transformed_positions = positions @ R.T
        return transformed_positions

    def transform_direction(self, phi):
        """
        旋转入射波的方向向量。

        参数：
        - phi：导线与入射波的夹角（以度为单位）
        """
        phi_rad = np.radians(phi)
        R = np.array([
            [np.cos(phi_rad), np.sin(phi_rad), 0],
            [-np.sin(phi_rad), np.cos(phi_rad), 0],
            [0, 0, 1]
        ])
        self.direction = R @ self.direction


if __name__ == '__main__':
    # z = np.linspace(-50, 50, 5000+1)
    # t = np.linspace(0, 100, 1000+1)
    a = np.pi / 6
    ground = Ground(mur=1, epr=3)
    nuc_source = NuclearExplosionSource(f=1e8, k=2.33, amplitude=50, alpha=0.5, beta=0.1, phi0=0, theta_i=0, ground=ground)

    def incide_wave(z, t):
        wave = []
        for i in z:
            _, wave_point, _, _ = nuc_source.propagate(a=a, t=[t], position=[0, 0, i])
            wave.append(wave_point)
        wave = np.array(wave)
        return wave

    def reflect_wave(z, t):
        wave = []
        for i in z:
            _, _, wavepoint, _ = nuc_source.propagate(a=a, t=[t], position=[0, 0, i])
            wave.append(wavepoint)
        E_direction = np.array([-nuc_source.r_parallel * np.cos(a) * np.cos(nuc_source.theta_i),
                                -nuc_source.r_perpendicular * np.sin(a),
                                -nuc_source.r_parallel * np.cos(a) * np.sin(nuc_source.theta_i)])
        wave = np.array(wave)
        sign = np.sign(wave @ E_direction)
        wave = np.array([np.linalg.norm(wave[i, :]) * sign[i] for i in range(len(wave))])
        return wave

    def hecheng_wave(z, t):
        wave = []
        for i in z:
            _, _, _, wavepoint = nuc_source.propagate(a=a, t=[t], position=[0, 0, i])
            wave.append(wavepoint)
        E_direction = np.array([-nuc_source.r_parallel * np.cos(a) * np.cos(nuc_source.theta_i),
                                -nuc_source.r_perpendicular * np.sin(a),
                                -nuc_source.r_parallel * np.cos(a) * np.sin(nuc_source.theta_i)])
        wave = np.array(wave)
        sign = np.sign(wave @ E_direction)
        wave = np.array([np.linalg.norm(wave[i, :]) * sign[i] for i in range(len(wave))])
        return wave

    def transmit_wave(z, t):
        wave = []
        for i in z:
            _, _, wavepoint = nuc_source.propagate(a=a, t=[t], position=[0, 0, i])
            wave.append(wavepoint)
        E_direction = np.array([nuc_source.t_parallel * np.cos(a) * np.cos(nuc_source.theta_i),
                                -nuc_source.t_perpendicular * np.sin(a),
                                nuc_source.t_parallel * np.cos(a) * np.sin(nuc_source.theta_i)])
        wave = np.array(wave)
        sign = np.sign(wave @ E_direction)
        wave = np.array([np.linalg.norm(wave[i, :]) * sign[i] for i in range(len(wave))])
        return wave

    m = [reflect_wave(z=[1], t=i) for i in np.linspace(0, 5e-8, 1000+1)]
    # plt.plot(np.linspace(0, 5e-8, 1000+1), m)
    # plt.show()

    # 初始化绘图
    fig, ax = plt.subplots()
    z1 = np.linspace(0, 25, 2500+1)
    z2 = np.linspace(-25, -0.01, 2500)

    # 多个波形，每个波形一个line对象
    line1, = ax.plot(z1, incide_wave(z1, 4.1e-8), label='incident')
    line2, = ax.plot(z1, reflect_wave(z1, 4.1e-8), label='reflect')
    line3, = ax.plot(z1, hecheng_wave(z1, 4.1e-8), label='hecheng')
    line4, = ax.plot(z2, transmit_wave(z2, 4.1e-8), label='transmit')
    lines = [line1, line2, line3, line4]

    ax.legend()
    plt.show()

    # 更新函数
    # def update(frame):
    #     lines[0].set_ydata(incide_wave(z1, frame))  # 更新每个波形的数据
    #     lines[1].set_ydata(reflect_wave(z1, frame))  # 更新每个波形的数据
    #     lines[2].set_ydata(hecheng_wave(z1, frame))  # 更新每个波形的数据
    #     lines[3].set_ydata(transmit_wave(z2, frame))  # 更新每个波形的数据
    #     return lines
    def update(frame):
        ax.cla()
        line1, = ax.plot(z1, incide_wave(z1, frame), label='incident')
        line2, = ax.plot(z1, reflect_wave(z1, frame), label='reflect')
        line3, = ax.plot(z1, hecheng_wave(z1, frame), label='hecheng')
        line4, = ax.plot(z2, transmit_wave(z2, frame), label='transmit')
        ax.legend()

    # 创建动画
    ani = FuncAnimation(fig, update, frames=np.linspace(4e-8, 5e-8, 100+1), interval=100, blit=False)
    plt.show()

    # _, s1 = nuc_source.propagate(a=np.pi / 6, position=[3, 0, 1])
    # _, s2 = nuc_source.propagate(a=np.pi / 6, position=[3, 0, -1])
    # E_direction = np.array([np.cos(np.pi / 6), -1, np.sin(np.pi / 6)])
    # sign1 = np.sign(s1 @ E_direction)
    # sign2 = np.sign(s2 @ E_direction)
    # plt.scatter(t, nuc_source.waveform, label='incidence')
    # plt.plot(t, np.array([np.linalg.norm(s1[i, :]) * sign1[i] for i in range(len(s1))]), label='reflect')
    # plt.plot(t, np.array([np.linalg.norm(s2[i, :]) * sign2[i] for i in range(len(s2))]), label='transmit')
    # plt.legend()
    #
    #
    # plt.show()




