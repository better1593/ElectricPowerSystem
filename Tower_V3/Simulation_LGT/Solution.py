import numpy as np

class Solution():
    def __init__(self):
        pass

    def Solution_V2(self, As, Rs, Ls, Cs, GLB, Soc_pos, i_sr):
        dt = GLB['dT']
        Nt = GLB['Nt']
        Nb, Nn = np.shape(As)
        SR_Is = np.zeros((Nn, Nt))
        SR_Vs = np.zeros((Nb, Nt))
        if Soc_pos:
            SR_Is[Soc_pos, :Nt] = i_sr  # source waveform and position
        else:
            SR_Vs = i_sr
        Nbs, Nns = np.shape(As)
        Ibts = np.zeros((Nbs, Nt))
        Vnts = np.zeros((Nns, Nt))
        Zlefts = Rs + Ls / dt
        Clefts = Cs / dt

        for kt in range(1, Nt):
            Vs = SR_Vs[:, kt]
            Is = SR_Is[:, kt]
            LEFTs = np.block([[-As, -Rs - Ls / dt], [Cs / dt, -As.T]])
            a = np.dot(Ls / dt, Ibts[:, kt - 1])
            b = np.dot(Clefts, Vnts[:, kt - 1])
            RIGHTs_Vs = Vs - np.dot(Ls / dt, Ibts[:, kt - 1])
            RIGHTs_Is = Is + np.dot(Clefts, Vnts[:, kt - 1])
            RIGHTs = np.concatenate((RIGHTs_Vs, RIGHTs_Is))
            outs = np.linalg.solve(LEFTs, RIGHTs)
            Ibts[:, kt] = outs[Nns:]
            Vnts[:, kt] = outs[:Nns]

        return Vnts, Ibts