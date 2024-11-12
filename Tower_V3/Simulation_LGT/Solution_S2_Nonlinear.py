import numpy as np

def Solution_S2_Nonlinear(As, Rs, Ls, Cs, Gs, GLB, Soc_pos, i_sr, SA_pos, SA_type, Ins_pos, In_type):
    dt = GLB['dT']
    Nt = GLB['Nt']
    Nb, Nn = As.shape
    SR_Is = np.zeros((Nn, Nt))
    SR_Vs = np.zeros((Nb, Nt))
    if Soc_pos:
        SR_Is[Soc_pos, :] = i_sr  # Source waveform and position
    else:
        SR_Vs[:, :] = i_sr
    Nbs, Nns = As.shape
    Ls = np.diag(np.diag(Ls) + 1e-12 * np.ones(Nbs))
    Rs = np.diag(np.diag(Rs) + 1e-9 * np.ones(Nbs))
    Cs = np.diag(np.diag(Cs))
    Gs = np.diag(np.diag(Gs))
    Ibts = np.zeros((Nbs, Nt))
    Vnts = np.zeros((Nns, Nt))
    R_nonlinear2 = np.zeros(Nbs)
    Rs0 = Rs
    Zlefts = Rs + Ls / dt
    Clefts = Gs + Cs / dt

    for kt in range(1, Nt):
        x0 = Ibts[SA_A0_pos, kt - 1]  # Current value at last moment
        temp_x0 = np.maximum(x0, 1e-6)
        y0 = (0.09 * np.log10(temp_x0) + 0.78) * 42.5e3  # Approximated V-I curve for A0
        R_nonlinear0 = y0 / temp_x0
        R_nonlinear2[SA_A0_pos] = R_nonlinear0

        x1 = Ibts[SA_A1_pos, kt - 1]  # Current value at last moment
        temp_x1 = np.maximum(x1, 1e-6)
        y1 = (0.08 * np.log10(temp_x1) + 0.61) * 42.5e3  # Approximated V-I curve for A1
        R_nonlinear1 = y1 / temp_x1
        R_nonlinear2[SA_A1_pos] = R_nonlinear1

        Rs[SA_A0_pos, SA_A0_pos] = np.diag(R_nonlinear0)
        Rs[SA_A1_pos, SA_A1_pos] = np.diag(R_nonlinear1)

        V = Vnts[Ins_pos, kt - 1]
        V_length = len(V)
        Dt = np.zeros(V_length)
        Ins_on = np.ones(V_length)

        V0 = 168.6e3  # Cao's paper
        DE_max = 140.4e3  # Cao's paper

        for ia in range(len(V)):
            if V[ia] >= V0:
                temp = (V[ia] - V0) * dt
                Dt[ia] += temp
                if Dt[ia] >= DE_max:
                    Ins_on[ia] = -1  # Switch close status of insulator

        R_Ins = 10 ** (Ins_on * 6)
        Rs[Ins_pos, Ins_pos] = np.diag(R_Ins)

        vs = SR_Vs[:, kt]
        is_ = SR_Is[:, kt]
        LEFTs = np.block([[-As, -Rs - Ls / dt], [Gs + Cs / dt, -As.T]])
        RIGHTs = np.concatenate((vs - Ls / dt @ Ibts[:, kt - 1], is_ + Cs / dt @ Vnts[:, kt - 1]), axis=1)
        outs = np.linalg.lstsq(LEFTs, RIGHTs, rcond=None)[0]
        Ibts[:, kt] = outs[Nns:]
        Vnts[:, kt] = outs[:Nns]

    return Vnts, Ibts, R_temp, Ins_on