import numpy as np
from scipy.optimize import minimize_scalar
from sympy import symbols, exp, solve


def Heidler(x):
    # Heidler Function
    # The possible values of N are limited to the integer values 2, 3 or 4.
    I0, t1, t2 = x[0], x[1], x[2]
    N = round(x[3])
    nm = exp((-t1 / t2) * (t2 * N / t1) ** (1 / N))
    # 将符号表达式转换为可供 minimize_scalar 使用的函数
    im2 = lambda tm: -((I0 / nm) * (((tm / t1) ** N) / (1 + (tm / t1) ** N)) * exp(-tm / t2).evalf())
    tm = symbols('tm')
    im = (I0 / nm) * (((tm / t1) ** N) / (1 + (tm / t1) ** N)) * exp(-tm / t2)

    # Ipc(Ipc对应的时间的范围是[0,50])!!!!!!!!!!!!
    res = minimize_scalar(im2, bounds=(0, 50), method='bounded')
    Ipc = abs(res.fun)  # 搜索过程中可能会出现复数
    # 搜索过程中可能会出现Ipc = []或Ipc=NaN
    if len(Ipc) == 0 or np.isnan(Ipc):
        Ipc = -100

    # tfc:time from 第一次10%*Ipc到第一次90%*Ipc
    tfc1_sol = solve(im - 0.1 * Ipc, tm)
    tfc2_sol = solve(im - 0.9 * Ipc, tm)
    tfc1 = np.min([float(sol) for sol in tfc1_sol])
    tfc2 = np.min([float(sol) for sol in tfc2_sol])
    tfc = abs(tfc2 - tfc1)
    # 搜索过程中可能会出现tfc = []
    if np.isnan(Ipc):
        tfc = -100

    #  thc：time from 第一次10%*Ipc到第二次50%*Ipc
    thc2_sol = solve(im - 0.5 * Ipc, tm)
    thc2 = np.max([float(sol) for sol in thc2_sol])
    thc = abs(thc2 - tfc1)
    # 搜索过程中可能会出现thc = []
    if np.isnan(Ipc):
        thc = -100

    return Ipc, tfc, thc