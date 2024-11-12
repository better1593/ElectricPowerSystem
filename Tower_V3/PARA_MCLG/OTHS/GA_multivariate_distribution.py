import numpy as np
from scipy.optimize import differential_evolution
from time import time
from Tower_V3.PARA_MCLG.OTHS.objhei import objhei

def GA_multivariate_distribution(parameterst):
    lightp = parameterst[:, 2:]
    light_final = np.zeros((len(parameterst), 5))

    for i in range(len(lightp)):
        #  for each quadruple of values for Ip , tf and th
        Ipi, tfi, thi = lightp[i, 0], lightp[i, 1], lightp[i, 3]
        # At first the values of c1, c2 and c3 are equal to each other.
        c1 = c2 = c3 = 1
        # 上下限的范围是[I0，t1,t2,N,Ip,tf,th]
        # I0是初始状态下的电流（Ip的范围是[3,200]）；t1，t2是信号传输线路中的时延或响应时间（tf的范围是[0.1,30] and
        # th的范围是[1,500]）
        bounds = [(1, 200), (0.1, 30), (1, 500), (2, 4), (3, 200), (0.1, 30), (1, 500)]
        tstart = time()
        # %画图 %并行计算,是否并行%显示每次迭代过程
        # The initial population size =50, The maximum number of generations= 100,
        # 遗传算法，未知数是I0，t1,t2,N,Ip,tf,th
        result = differential_evolution(
            objhei(),
            bounds,
            args=(c1, c2, c3, Ipi, tfi, thi, tstart),
            strategy='best1bin',
            maxiter=100,
            popsize=50,
            tol=0.01,
            mutation=(0.5, 1),
            recombination=0.7,
            seed=None,
            callback=None,
            disp=True,
            polish=True,
            init='latinhypercube',
            atol=0
        )

        # 输出最终解
        x = result.x
        # The possible values of N are limited to the integer values 2, 3 or 4.
        I0, t1, t2, N = x[0], x[1], x[2], round(x[3])
        nm = np.exp((-t1 / t2) * (t2 * N / t1) ** (1 / N))
        Ipc, tfc, thc = x[4], x[5], x[6]

        light_final[i, :] = [I0, nm, t1, N, t2]

    return light_final