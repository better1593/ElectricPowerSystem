import numpy as np
import warnings
import Vector_Fitting.Drivers.RP_Driver as RPdriver
import Drivers.VF_Driver as VFdriver
import scipy.io

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

    s = 1j * 2 * np.pi * f0
    poles = []  # Initial poles are automatically generated
    SER, *_ = VFdriver.drive(Zi, s, poles, VFopts)

    RFopts = {'Niter_in': 5}
    SER, Zfit, *_ = RPdriver.drive(SER, s, RFopts)

    R0 = SER['D']
    L0 = SER['E']

    Nc = Zi.shape[0]
    Ln = np.zeros((Nc, Nc, VFopts['N']))
    Rn = np.zeros((Nc, Nc, VFopts['N']))

    for ik in range(VFopts['N']):
        Rn[:, :, ik] = SER['R'][:, :, ik] / SER['poles'][ik]
        Ln[:, :, ik] = -1 / SER['poles'][ik] * Rn[:, :, ik]

    return R0, L0, Rn, Ln, Zfit


if __name__ == "__main__":
    mat_contents = scipy.io.loadmat(r'D:\Documents\a实验室\过电压计算程序\Electronic_model_building-main\Vector_Fitting-main\Data\input.mat')
    Zi = mat_contents['Zi']
    f0 = mat_contents['f0']
    Nfit = 9
    R0, L0, Rn, Ln, Zfit = vecfit_kernel_Z_Ding(Zi, f0, Nfit)
    print("start")