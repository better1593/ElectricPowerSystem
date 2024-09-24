from scipy.io import loadmat
import numpy as np
from Vector_Fitting.Drivers.VFdriver import VFdriver
from Vector_Fitting.Drivers.RPdriver import RPdriver
from Vector_Fitting.Calculators.plots import plot_figure_11
from Vector_Fitting.Calculators.create_netlist import create_netlist_file

bigY = loadmat('..//Data/impedance.mat')['impedance_matrix'][:, :, :19]
s = np.squeeze(loadmat('../Data/s.mat')['s'])
poles = loadmat('../Data/poles.mat')['poles']
#
# bigY = loadmat('..//Data/Zi2.mat')['Zi'][:, :, :19]
# s = np.squeeze(loadmat('../Data/s2.mat')['s'])
# poles = loadmat('../Data/poles2.mat')['poles']

# bigY = loadmat('Data/Zi3.mat')['Zi']
# s = np.squeeze(loadmat('Data/s3.mat')['s'])
# poles = loadmat('Data/poles2.mat')['poles']

# Pole-Residue Fitting
vf_driver = VFdriver(N=9,
                     poletype='lincmplx',
                     weightparam='common_1',
                     Niter1=10,
                     Niter2=5,
                     asymp='DE',
                     plot=False
                     )
poles=None
SER, *_ = vf_driver.vfdriver(bigY, s, poles)

# plot_figure_11(s, bigY, bigYfit, SER)

# Passivity Enforcement
rp_driver = RPdriver(parametertype='y',
                     Niter_in = 5,
                     plot=False,
                     # s_pass=2*np.pi*1j*np.linspace(0, 2e5, 1001).T,
                     # ylim=np.array((-2e-3, 2e-3))
                     )
SER, bigYfit_passive, opts3 = rp_driver.rpdriver(SER, s)

plot_figure_11(s, bigY, bigYfit_passive, SER)

poles = SER['poles']
residues = SER['C']
Ns = poles.shape[0]
Nc = int(residues.shape[1] / Ns)
poles = poles.reshape((1, -1))
residues = residues.reshape((Nc ** 2, Ns))
create_netlist_file(poles, residues)