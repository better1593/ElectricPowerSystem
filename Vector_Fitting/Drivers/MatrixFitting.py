import warnings
import numpy as np

from Vector_Fitting.Drivers.VFdriver import VFdriver
from Vector_Fitting.Drivers.RPdriver import RPdriver
from Vector_Fitting.Calculators.plots import plot_figure_11

def matrix_vector_fitting(impedance_matrix, varied_frequency, Nfit=9):
    """
    【函数功能】 矩阵矢量拟合 vector fitting
    :param impedance_matrix: numpy.ndarray, n*n*nf 阻抗矩阵
    :varied_frequency: numpy.ndarray, 1*nf 频率矩阵

    【出参】
    :return: dict 拟合结果
    """
    s = 1j * 2 * np.pi * varied_frequency
    vf = VFdriver(poletype='lincmplx', weightparam='common_1', plot=False, asymp='DE', N=Nfit, Niter1=10, Niter2=5)
    SER, *_ = vf.vfdriver(impedance_matrix, s, None)

    rp = RPdriver(Niter_in=5, plot=False, parametertype='y')
    SER, Zfit, *_ = rp.rpdriver(SER, s)

    # plot_figure_11(2j * np.pi * varied_frequency, impedance_matrix, Zfit,
    #                SER)

    return SER