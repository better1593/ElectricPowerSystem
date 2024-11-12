from time import time
from Tower_V3.PARA_MCLG.Heidler import Heidler

def objhei(x, c1, c2, c3, Ipi, tfi, thi, tstart):
    Ipc, tfc, thc = Heidler(x)
    I0 = x[0]
    t1 = x[1]
    t2 = x[2]
    N = round(x[3])
    x[4] = Ipc
    x[5] = tfc
    x[6] = thc

    # If after some tens of attempts, conditions (13) are still not satisfied,
    # then the time to half value is penalized by means of a reduction of c3 with respect to c1 and c2.
    # 假设c3是随着运行次数的增加而逐渐减小,而不是一次性减小并在后续尝试中保持不变
    if time() - tstart >= 5 * 60:
        c3 = c3 * 0.9 ** ((time() - tstart) / (5 * 60))

    # The algorithm is stopped if the relative errors on the three parameters satisfy all the three following conditions:
    # the conditions of (13)
    if abs((Ipc - Ipi) / Ipi) < 0.5e-2 and abs((tfc - tfi) / tfi) < 0.5e-2 and abs((thc - thi) / thi) < 1e-2:
        f = -1e10
    else:
        f = c1 * abs((Ipc - Ipi) / Ipi) + c2 * abs((tfc - tfi) / tfi) + c3 * abs((thc - thi) / thi)

    return f