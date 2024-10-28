import numpy as np
import matplotlib.pyplot as plt


class NuclearExplosionSource:
    def __init__(self, k, amplitude, alpha, beta, incident_angle):
        """
        初始化核爆源模型。

        参数：
        - k: 脉冲幅值校正系数
        - amplitude：波形的峰值幅度
        - alpha：第一个指数项的衰减常数
        - beta：第二个指数项的衰减常数
        - incident_angle：入射角，单位为°
        """
        self.k = k
        self.amplitude = amplitude
        self.alpha = alpha
        self.beta = beta
        self.incident_angle = np.radians(incident_angle)
        # 根据入射角计算方向向量（假设方位角对称）
        self.direction = np.array([
            np.sin(self.incident_angle),
            0,
            np.cos(self.incident_angle)
        ])

    def waveform(self, t):
        """
        计算波形
        """
        s = -self.k * self.amplitude * (np.exp(-self.alpha * t) - np.exp(-self.beta * t))
        return s  # 对于 t < 0，波形为零

    def propagate(self, times, positions):
        """
        根据平面波公式计算某位置某时刻的波形
        """
        c = 3e8  # 传播速度（例如，真空中的光速）
        distances = np.linalg.norm(positions - self.initial_position, axis=1)
        propagation_delays = distances / c
        waveforms = []

        for idx, pos in enumerate(positions):
            t_effective = times - propagation_delays[idx]
            s = self.waveform(t_effective)
            waveforms.append(s)

        return np.array(waveforms)

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
    nuc_source = NuclearExplosionSource(k=2.33, amplitude=50, alpha=0.5, beta=0.1, incident_angle=np.pi/4)
    t = np.linspace(0, 1000, 1000)
    s = nuc_source.waveform(t)
    plt.plot(t, s)
    plt.show()




