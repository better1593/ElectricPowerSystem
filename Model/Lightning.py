import math
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False
class StrokeParameters:
    CIGRE_PARAMETERS = {
        '0.25/100us': [0.25, 100, 6.4, 2, 0.9, 30, 0.5, 80, 2, 0.7],
        '8/20us': [8, 20, 1.18, 2.6, 0.93, 19.9, 0.51, 50, 1.5, 0.4],
        '2.6/50us': [2.6, 50, 2.56, 2.1, 0.92, 30, 0.5, 80, 1.8, 0.6],
        '10/350us': [10, 350, 0.92, 2.1, 0.98, 45, 0.53, 160, 2, 0.7]
    }

    HEIDLER_PARAMETERS = {
        '0.25/100us': [0.9, 0.25, 100, 2],
        '0.25/2.5us': [10.7e3, 0.25, 2.5, 2],
        '8/20us': [30.85, 8, 20, 2.4],
        '2.6/50us': [10e5, 2.6, 50, 2.1],
        '10/350us': [44.43, 10, 350, 2.1]
    }

    CHANNELMODEL_PARAMETERS = {
        'TL': [0, 0, 2.3077],
        'MTLL': [7000, 0, 2.3077],
        'MTLE': [0, 1700, 2.3077]
    }


class Channel:
    def __init__(self, hit_pos, model='MTLE'):
        # 雷电通道相关参数
        self.height = 1000  # 雷电通道高度
        self.dh = 10  # 通道段长度
        self.N_channel_segment = int(self.height / self.dh)  # 雷电通道被划分后的个数
        self.hit_pos = hit_pos  # 雷击点的坐标
        self.channel_model = model  # 模型
        # 与电磁场计算的相关常数，根据模型的不同取不同的值
        self.H = StrokeParameters.CHANNELMODEL_PARAMETERS[model][0]
        self.lamda = StrokeParameters.CHANNELMODEL_PARAMETERS[model][1]
        self.vcof = 1 / StrokeParameters.CHANNELMODEL_PARAMETERS[model][2]


class Stroke:
    def __init__(self, stroke_type: str, duration: float, dt: float, is_calculated: bool, parameter_set: str, parameters=None):
        """
        初始化脉冲对象

        参数说明:
        stroke_type (str): 脉冲的类型, 目前只支持 'CIGRE' 和 'Heidler'
        stroke_sequence (int): 回击的序号
        duration (float): 脉冲持续时间，单位s
        dt (float): 仿真步长，单位s
        is_calculated (bool): 脉冲是否要被计算
        parameter_set (str): 脉冲参数集, 目前只支持 '0.25/100us', '8/20us', '2.6/50us', '10/350us'
        parameters (list, optional): 脉冲参数, 仅在 'CIGRE' 和 'Heidler' 类型时使用, parameter_set被指定时, 请勿初始化该参数, 如想测试parameter_set之外的参数集, 请在此处初始化参数列表
        """
        self.stroke_type = stroke_type
        self.duration = duration
        self.is_calculated = is_calculated
        self.dt = dt  # 时间步长
        # self.Nt = 1000  # 时间步数
        self.Nt = int((self.duration / self.dt))  # 每个stroke电流的最大采样点数
        # self.N_max = 200000  # 最大采样点数


        self.t_us = np.array(list(range(self.Nt))) * self.dt # 时刻，单位为us
        # self.dt = self.duration / self.Nt
        self.current_waveform = []  # 雷电流波形的时间序列

        # self.stroke_interval = 1.0e-3  # 每个stroke的间隔为1ms

        # parameter_set与parameters二选一传入，最终决定参数列表归属
        if parameter_set:
            if stroke_type == 'CIGRE':
                self.parameters = StrokeParameters.CIGRE_PARAMETERS[parameter_set]
            elif stroke_type == 'Heidler':
                self.parameters = StrokeParameters.HEIDLER_PARAMETERS[parameter_set]
            else:
                raise ValueError("Invalid stroke type. Must be 'CIGRE' or 'Heidler'.")

        if parameters:
            self.parameters = parameters

    def cigre_waveform(self, t):
        tn, A, B, n, I1, t1, I2, t2, Ipi, Ipc = self.parameters
        tn = tn * 1.0e-6
        t1 = t1 * 1.0e-6
        t2 = t2 * 1.0e-6
        # 初始化电流波形
        Iout = np.zeros(self.Nt)
        Iout1 = A * t[t <= tn] + B * t[t <= tn] ** n
        Iout2 = I1 * np.exp(-(t[t > tn] - tn) / t1) - I2 * np.exp(-(t[t > tn] - tn) / t2)
        Iout[t <= tn] = Iout1
        Iout[t > tn] = Iout2

        Iout *= Ipi / Ipc
        return Iout

    def heidler_waveform(self, t):
        Ip, tau1, tau2, n = self.parameters
        tau1 = tau1 * 1.0e-06
        tau2 = tau2 * 1.0e-06
        eta = math.exp(-(tau1 / tau2) * ((n * tau2 / tau1) ** (1 / n)))
        # eta = 1
        Iout = ((Ip / eta) * ((t / tau1) ** n) / (1 + (t / tau1) ** n)) * np.exp(-t / tau2)

        return Iout

    def add_cos_window_to_waveform_tail(self, current_waveform):
        """
        给电流波形的尾部应用余弦窗，平滑波形
        """
        # 计算尾部应用余弦窗的点数, n的取值与采样点个数和间隔时间有关，后续待修改
        Ntail = int(np.floor((self.duration / 10) / self.dt))
        # 生成theta值
        theta = np.linspace(0, np.pi, Ntail, endpoint=False)
        # 计算余弦窗
        coswin = 0.5 * np.cos(theta) + 0.5
        # 应用余弦窗
        current_waveform[-Ntail:] *= coswin
        return current_waveform

    def calculate(self):
        """
        Calculate the pulse waveform at the given time series.

        Args:
            t (float): Time in seconds.

        Returns:
            float: The value of the pulse waveform at the given time series.
        """
        # Calculate only when is_calculated==True
        if self.is_calculated:
            if self.stroke_type == 'CIGRE':
                # self.current_waveform = self.add_cos_window_to_waveform_tail(self.cigre_waveform(self.t_us))
                self.current_waveform = self.cigre_waveform(self.t_us)
                return self.current_waveform
            elif self.stroke_type == 'Heidler':
                # self.current_waveform = self.add_cos_window_to_waveform_tail(self.heidler_waveform(self.t_us))
                self.current_waveform = self.heidler_waveform(self.t_us)
                return self.current_waveform
        return 0


class Lightning:
    def __init__(self, id: int, type: str, strokes, channel):
        """
        初始化雷电对象

        参数说明:
        id (int): 雷电序号
        type (str): 雷电类型, 请指定'Direct'或'Indirect', 用以表示直击雷或间接雷
        strokes (list, optional): 雷电的脉冲列表, 如未指定, 默认为空列表
        """
        self.id = id
        self.type = type
        self.strokes = strokes or []
        self.stroke_number = len(self.strokes)
        self.channel = channel


    def add_stroke(self, stroke: Stroke):
        self.strokes.append(stroke)
        self.stroke_number = len(self.strokes)

    def total_waveform(self):
        """
        Calculate the total lightning waveform at the given time.

        Args:
            t (float): Time in seconds.

        Returns:
            float: The value of the total lightning waveform at the given time.
        """
        total = []
        for stroke in self.strokes:
            total.append(stroke.calculate())
        total = np.concatenate(total, axis=0)
        return total


if __name__ == '__main__':
    stroke1 = Stroke('Heidler', duration=2.0e-5, dt=1.0e-8, is_calculated=True, parameter_set='0.25/2.5us',
                    parameters=None)
    # stroke2 = Stroke('Heidler', duration=1e-3, is_calculated=True, parameter_set='2.6/50us',
    #                 parameters=None)
    strokes = [stroke1]
    channel = Channel(hit_pos=[50, 500, 0])
    lightning = Lightning(id=1, type='Indirect', strokes=strokes, channel=channel)
    t = []
    last_stroke_duration = 0
    for stroke in lightning.strokes:
        t_us = stroke.t_us + last_stroke_duration
        t.append(t_us)
        last_stroke_duration = stroke.duration
    t = np.concatenate(t, axis=0)
    I = lightning.total_waveform()
    plt.xlabel('时间 (us)')
    plt.ylabel('电流 (kA)')
    plt.plot(t, I)
    plt.show()