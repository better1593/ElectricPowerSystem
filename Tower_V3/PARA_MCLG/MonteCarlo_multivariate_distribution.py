from concurrent.futures import ProcessPoolExecutor
import numpy as np


def MonteCarlo_multivariate_distribution (parameterst):
    light_final = []
    with ProcessPoolExecutor() as executor:
        results = Compute_Light_Final(parameterst)
        for result in results:
            light_final.append(result)
    return light_final

def Compute_Light_Final(params):
    light_final = []
    a = params.shape[0]
    for i in range(0, params.shape[0] + 1):
        # for each quadruple of values for Ip , tf and th
        # 换算单位,实际Ipi-kA-A，tfi-μs-s，Smi-kA/μs-A/s，thi-μs-s
        Ipi, tfi, Smi, thi = params[i - 1, 2] * 1e3, params[i - 1, 3] * 1e-6, params[i - 1, 4] * (1e3 / 1e-6), \
                             params[i - 1, 5] * 1e-6

        SN = Smi * tfi / Ipi
        n = 1 + 2 * (SN - 1) * (2 + 1 / SN)
        # In case a Monte Carlo event presents a value of n out of these bounds, the value of Sm is adjusted as
        if n < 1:
            Smi = 1.01 * Ipi / tfi
            SN = Smi * tfi / Ipi
            n = 1 + 2 * (SN - 1) * (2 + 1 / SN)
        elif n > 55:
            Smi = 12 * Ipi / tfi
            SN = Smi * tfi / Ipi
            n = 1 + 2 * (SN - 1) * (2 + 1 / SN)

        tn = 0.6 * tfi * (3 * SN ** 2 / (1 + SN ** 2))
        A = (1 / (n - 1)) * (0.9 * (Ipi / tn) * n - Smi)
        B = (1 / ((tn ** n) * (n - 1))) * (Smi * tn - 0.9 * Ipi)
        t1, t2 = (thi - tn) / np.log(2), 0.1 * Ipi / Smi
        # 按照文章编写公式错误，细心
        I1 = ((t1 * t2) / (t1 - t2)) * (Smi + 0.9 * (Ipi / t2))
        I2 = ((t1 * t2) / (t1 - t2)) * (Smi + 0.9 * (Ipi / t1))

        # 求解最大值Ipc(Ipc对应的时间的范围是[0,50]),单位是μs!!!
        # 缩小搜索范围，使速度更快
        lb, ub = (tn / 10 ** (np.log10(tn)) * 10 - 5) * (10 ** (np.log10(tn) - 1)), \
                 (tn / 10 ** (np.log10(tn)) * 10 + 5) * (10 ** (np.log10(tn) - 1))
        # 第2种方法：用时4.734314 秒，精度与第1种方法差不多
        step = 10 ** (np.log10(lb) - 3)
        num = int((ub - lb) / step) + 1
        t = np.linspace(lb, ub, num)
        y = np.where(t <= tn, A * t + B * (t ** n), I1 * np.exp(-(t - tn) / t1) - I2 * np.exp(-(t - tn) / t2))
        # 找到 y 中的最大值及其对应的索引
        Ipc = np.max(y)

        # As this procedure can lead to small errors on the resulting
        # current peak, the current is normalized to the desired peak value.
        # syms t
        # Cigre Function
        y2 = lambda t: ((t <= tn) * (A * t + B * (t ** n)) + (t > tn) * (
                I1 * np.exp(-(t - tn) / t1) - I2 * np.exp(-(t - tn) / t2))) * (Ipi / Ipc)

        # lighting parameters:tn A B n I1 t1 I2 t2 Ipi Ipc

        new_row = [tn, A, B, n, I1, t1, I2, t2, Ipi, Ipc]
        light_final.append(new_row)
    return light_final