import numpy as np
import os
import sys
import shutil
import pandas as pd
import ast

from Tower_V3.PARA_MCLG.MC_Init import MC_Init
from Tower_V3.PARA_MCLG.MC_Init_S import MC_Init_S
from Tower_V3.PARA_MCLG.MC_Data_Gene import MC_Data_Gene
from Tower_V3.PARA_MCLG.MC_Data_Gene_S import MC_Data_Gene_S
from Tower_V3.Simulation_MCLGT.LGTMC_Solu import LGTMC_Solu
import re

class Simulation_MC_LGT():
    def __init__(self, FDIR):
        self.FDIR = FDIR  # Initialize the Simulation_LGT class with the given file directory paths

    def Simulation_MC_LGT(self,Tower, Span, Cable, GLB, LGT):
        # GLB['SSdy']['flag'] = [1, 0, 0, 0, 0, 0, 0] 没有GLB.SSdy
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!注释了！！！！！！！！！！！！！
        [LatDis, WaveModel, MC_lgtn, DSave,AR] = MC_Init() # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        filepath = GLB['FDIR']['dataMCLG']
        if os.path.exists(filepath):
            shutil.rmtree(filepath)

        os.mkdir(filepath)


        [EdgeXY, Icurr, Summary, StrokePosi, Iwave, FSdist, PoleXY] \
            = MC_Data_Gene.MC_Data_Gene(self, GLB, Span, Tower, MC_lgtn, DSave,LatDis, WaveModel,AR)

        excel_path = filepath + '/' + 'Pole_XY.xlsx'
        PoleXY = pd.read_excel(excel_path,).to_numpy()
        excel_path = filepath + '/' + 'Flash_Stroke Dist.xlsx'
        FSdist = pd.read_excel(excel_path).to_numpy()
        excel_path = filepath + '/' + 'Current Waveform_CIGRE.xlsx'
        Iwave = pd.read_excel(excel_path).to_numpy()
        excel_path = filepath + '/' + 'Current Parameter.xlsx'
        Icurr = pd.read_excel(excel_path).to_numpy()
        excel_path = filepath + '/' + 'Stroke Position.xlsx'
        StrokePosi2 = pd.read_excel(excel_path).to_numpy()
        StrokePosi = np.empty((StrokePosi2.shape[0], StrokePosi2.shape[1] + 2), dtype=object)
        StrokePosi[:, 0:7] = StrokePosi2[:, 0:7]
        for ist in range(StrokePosi2.shape[0]):
            cleaned_string =  StrokePosi2[ist, 7].replace('np.float64(', '').replace(')', '')
            try:
                St1a = ast.literal_eval(cleaned_string)
                StrokePosi[ist, 7:9] = np.array(St1a)
            except ValueError as e:
                print(f"Error: {e}")

            if (StrokePosi2[ist, 8])=='[nan, nan]':
                StrokePosi[ist, 9:11] = np.nan*2
            else:
                cleaned_string = StrokePosi2[ist, 8].replace('np.float64(', '').replace(')', '')
                try:
                    St1b = ast.literal_eval(cleaned_string)
                    StrokePosi[ist, 9:11] = np.array(St1b)
                except ValueError as e:
                    print(f"Error: {e}")


        MCLGT = {}
        MCLGT['Num'] = len(FSdist)

        MCLGT['flag'] = 1
        MCLGT['huri'] = []
        MCLGT['radi'] = 60
        MCLGT['tsel'] = 0

        FO_Type = 1
        GLB['FOtype'] = FO_Type

        output = LGTMC_Solu(Tower, Span, Cable, GLB, LGT, MCLGT,
        Icurr, Iwave, StrokePosi, FSdist, PoleXY)

        return output
