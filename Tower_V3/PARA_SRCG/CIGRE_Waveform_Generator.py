import numpy as np

def CIGRE_Waveform_Generator(WavePara, N0, dT):
    """
    Iout = ((t<= tn).* (A*t+B*(t^n)) + (t>tn) .* (I1*exp(-(t-tn)/t1) - I2*exp(-(t-tn)/t2)))*(Ipi/Ipc);
    """
    tn = WavePara[0]
    A = WavePara[1]
    B = WavePara[2]
    n = WavePara[3]
    I1 = WavePara[4]
    t1 = WavePara[5]
    I2 = WavePara[6]
    t2 = WavePara[7]
    Ipi = WavePara[8]
    Ipc = WavePara[9]

    t = np.arange(0, N0) * dT
    T0_le_tn = t[t <= tn]  # index for t=<tn
    Iout1 = A * T0_le_tn + B * (T0_le_tn ** n)
    T0_gt_tn = t[t > tn]   # index for t=<tn
    Iout2 = I1 * np.exp(-(T0_gt_tn - tn) / t1) - I2 * np.exp(-(T0_gt_tn - tn) / t2)

    Iout = np.concatenate([Iout1, Iout2]) * (Ipi / Ipc)

    return Iout