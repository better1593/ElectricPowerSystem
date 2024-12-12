import numpy as np
import warnings
from Vector_Fitting.Drivers.RPdriver import RPdriver
from Vector_Fitting.Drivers.VFdriver import VFdriver
from Vector_Fitting.Calculators.plots import plot_figure_11
import scipy.io
# 40 41行修改
def vecfit_kernel_Z_Ding(Zi, f0, Nfit, vf_mod=None):
    """
    Vector Fitting Toolkit in Python
    :param Zi: Impedance matrix (n, n, Ns)
    :param f0: Frequency samples (1, Ns)
    :param Nfit: Number of poles to fit
    :param vf_mod: Optional model selection for asymptotic behavior
    :return: R0, L0, Rn, Ln, Zfit
    """

    if f0.size < 2:
        warnings.warn('Parameters for Vector Fitting MUST be Multi-Frequency', UserWarning)
        R0 = np.real(Zi)
        L0 = np.imag(Zi) / (2 * np.pi * f0)
        return R0, L0, None, None, None

    VFopts = {'asymp': vf_mod if vf_mod is not None else 3, 'plot': 0, 'N': Nfit, 'Niter1': 10, 'Niter2': 5}

    s = np.squeeze(1j * 2 * np.pi * f0)
    poles = None  # Initial poles are automatically generated

    # Pole-Residue Fitting
    vf_driver = VFdriver(N=Nfit,
                         poletype='lincmplx',
                         weightparam='common_1',
                         Niter1=10,
                         Niter2=5,
                         asymp='DE' if vf_mod is None else vf_mod,
                         plot=False
                         )
    SER, _, Zfit = vf_driver.vfdriver(Zi, s, poles)
    # 绘制误差曲线
    diff2 = Zfit - Zi
    error2 = np.sqrt(np.sum(np.abs(diff2 ** 2), axis=2))
    plot_figure_11(s, Zi, Zfit, SER)

    # Passivity Enforcement
    rp_driver = RPdriver(parametertype='y',
                         # TOLGD=1e-3,
                         # Niter_out=20,
                         Niter_in=10,
                         # weightfactor=0.01,
                         # complx_ss=False,
                         # WeightParam='indiv_sqrt',
                         plot=False
                         # s_pass=2*np.pi*1j*np.linspace(0, 2e5, 1001).T,
                         # ylim=np.array((-2e-3, 2e-3))
                         )
    SER, Zfit, *_ = rp_driver.rpdriver(SER, s)

    # 被动性抑制后再次绘制误差
    diff = Zfit - Zi
    error = np.sqrt(np.sum(np.abs(diff ** 2), axis=2))
    plot_figure_11(s, Zi, Zfit, SER)

    R0 = np.real(SER['D'])  # 只取实部
    L0 = np.real(SER['E']) if vf_mod is None else None  # 只取实部

    Nc = Zi.shape[0]
    Ln = np.zeros((Nc, Nc, VFopts['N']))
    Rn = np.zeros((Nc, Nc, VFopts['N']))

    for ik in range(VFopts['N']):
        Rn[:, :, ik] = np.real(SER['R'][:, :, ik] / SER['poles'][ik])  # 只取实部
        Ln[:, :, ik] = np.real(-1 / SER['poles'][ik] * Rn[:, :, ik])  # 只取实部

    return R0, L0, Rn, Ln, Zfit
