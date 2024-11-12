import pandas as pd
import numpy as np

def MC_Init_S():

    # 从表格中提取数值
    filename = 'D:\python_work\Tower_V3\PARA_MCLG\MC_InitInput_S.xlsx'
    inp1 = pd.read_excel(filename, sheet_name='Sheet1', index_col=0)
    inp2 = pd.read_excel(filename, sheet_name='Sheet2', index_col=0)
    inp3 = pd.read_excel(filename, sheet_name='Sheet3', index_col=0)
    inp4 = pd.read_excel(filename, sheet_name='Sheet4', index_col=0)
    inp5 = pd.read_excel(filename, sheet_name='Sheet5', index_col=0)

    # 更新 MC_lgtn 和其他变量
    LatDis_max = inp1.iloc[0, 0]
    Wave_Model = inp1.iloc[1, 0]
    MC_lgtn={
        'mode': inp1.iloc[2, 0],
        'fixn': inp1.iloc[3, 0],
        'stroke': inp1.iloc[4, 0],
        'averageh': inp1.iloc[5, 0],
        'muN': inp2.iloc[0, 0],
        'sigmaN': inp2.iloc[0, 1],
        'lmuI': inp3.iloc[:, 0].values,
        'sigma1st': inp3.iloc[:, 1].values,
        'lmu': inp4.iloc[:, 0].values,
        'sigma': inp4.iloc[:, 1].values,
        'rhoi': inp5.values
    }

    MC_lgtn['muI'] = np.log(MC_lgtn['lmuI'])
    MC_lgtn['mu'] = np.log(MC_lgtn['lmu'])

    # 初始化 DSave 字典
    DSave = {
        'userInput0': 1,
        'userInput1': 1,
        'userInput2': 1,
        'userInput3': 1,
        'userInput3d': 1,
        'userInput3ind': 1,
        'userInput4': 1,
        'userInput4d': 1,
        'userInput4ind': 1
    }

    return LatDis_max, Wave_Model, MC_lgtn, DSave


