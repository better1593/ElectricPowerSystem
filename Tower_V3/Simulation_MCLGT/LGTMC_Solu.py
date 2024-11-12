import numpy as np

from Tower_V3.Simulation_MCLGT.Sim_MCLGT_Init import Sim_MCLGT_Init
from Tower_V3.Simulation_MCLGT.Huri_Method import Huri_Method



def LGTMC_Solu (Tower, Span, Cable, GLB, LGT, MCLGT,
                Icurr, Iwave, StrPosi, FSdist, PoleXY):

    Flash_Num = MCLGT['Num']

    for id in range(Flash_Num):
        GLB, LGT = Sim_MCLGT_Init (Icurr, Iwave, StrPosi, FSdist, GLB, LGT, id)
        flash = LGT['Soc']['flash']
        StrNum = flash['head'][1]

        if FSdist [id, 2] == 1 and (MCLGT['tsel'] == -1 or MCLGT['tsel'] == 1):
            output = LGT1_Solu (Tower, Span, Cable, GLB, LGT)

        if FSdist [id, 2] == 0 and (MCLGT['tsel'] == -1 or MCLGT['tsel'] == 0):
            PoleApp = []
            StrPos = LGT['Soc']['pos'][2:4]
            if MCLGT['huri']:
                for jd in range(StrNum):
                    icur = np.concatenate((flash['para'][jd, 2:6], StrPos))
                    result_huri = Huri_Method (MCLGT, icur)
                    if result_huri == 1:
                        if jd == 0 or not PoleApp:
                            dist = np.sqrt((icur[4] - PoleXY[:, 1])**2 + (icur[5] - PoleXY[:, 2])**2)
                            dex = np.argsort(dist)
                            dmin = dist[dex]
                            PoleApp = [PoleXY[dex[0], 1:3], PoleXY[dex[1], 1:3], dmin[0], dmin[1]]
                        MCLGT['huri'].append(np.concatenate((icur, PoleApp)))
                        flash['flag'][0, jd] = 0
                LGT['Soc']['flash'] = flash
                GLB['Soc']['flash'] = flash

                output = LGT1_Solu (Tower, Span, Cable, GLB, LGT)

                tmp1 = output[0]['FO'] * flash['flag']
                dex1 = np.where(tmp1 > 0)[0]
                if not PoleApp:
                    dist = np.sqrt((StrPos[0] - PoleXY[:, 1])**2 + (StrPos[1] - PoleXY[:, 2])**2)
                    dex = np.argsort(dist)
                    dmin = dist[dex]
                    PoleApp = [PoleXY[dex[0], 1:3], PoleXY[dex[1], 1:3], dmin[0], dmin[1]]
                tmp2 = np.tile(np.concatenate((StrPos, PoleApp)), (len(dex1), 1))
                MCLGT['huri'].append(np.concatenate((flash['para'][dex1, 2:6], tmp2)))

    return output