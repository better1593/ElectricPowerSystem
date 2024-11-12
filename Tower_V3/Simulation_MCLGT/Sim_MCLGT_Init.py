import numpy as np
from Tower_V3.PARA_SRCG.Source_Current_Generator import Source_Current_Generator

def Sim_MCLGT_Init (Icurr, Iwave, StrPosi, FSdist, GLB, LGT, id):

    dT = GLB['dT']
    T0 = GLB['T0']
    Nt = GLB['Nt']
    N0 = GLB['N0']
    Nmax = GLB['Nmax']
    Lch = LGT['Lch']


    flag = [1, 0, 0, 0, 0, 0, 0]
    SSdy = {
        'flag': flag,
        'Nca': 1,
        'dat': []
    }

    StrNum = FSdist[id, 1]
    WavMod = 1
    Ipos = np.where(Icurr[:, 0] == (id + 1))[0],
    flash = {
        'head': np.hstack((FSdist[id, 0:2], WavMod, 0)),
        'flag': [1] * StrNum,
        'para': Icurr[Ipos],
        'wave': Iwave[Ipos]
    }

    Icur = Source_Current_Generator (flash, T0, N0, dT)

    Soc = {
        'typ': FSdist[id, 2],
        'dat': Icur,
        'flash': flash
    }
    if Soc['typ'] == 1:
        Soc['pos'] = StrPosi[Ipos[0][0]][1:7]
    elif Soc['typ'] == 0:
        Soc['pos'] = np.hstack((0,StrPosi[Ipos[0]][0][7:9]))

    Lch = {
        'curr': [],
        'pos': StrPosi[Ipos[0][0]][7:9]
    }

    LGT['Soc'] = Soc
    LGT['SSdy'] = SSdy
    LGT['Lch'] = Lch
    GLB['Soc'] = Soc
    GLB['SSdy'] = SSdy

    return GLB, LGT
