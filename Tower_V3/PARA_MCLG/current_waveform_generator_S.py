import numpy as np
import os
import pandas as pd
from Tower_V3.PARA_MCLG.MonteCarlo_multivariate_distribution import MonteCarlo_multivariate_distribution
from Tower_V3.PARA_MCLG.OTHS.GA_multivariate_distribution import GA_multivariate_distribution

def current_waveform_generator_S (allp,Wave_Model,DSave,resultstro,resultcur,foldname):
    # struct获取每个变量名
    way = Wave_Model
    userInput4 = DSave['userInput4']
    userInput4d = DSave['userInput4d']
    userInput4ind = DSave['userInput4ind']
    sited = [int(x) for x in resultstro['sited']]
    siteind = resultstro['siteind']
    parameterst = resultcur['parameterst']

    if way == 1:
        light_final = MonteCarlo_multivariate_distribution(parameterst)

    if way == 2:
        light_final = GA_multivariate_distribution (parameterst)

    # save
    if userInput4 == 1 and way == 1:
        # 将double 数组转换为table
        df4 = pd.DataFrame(light_final, columns=['tn', 'A', 'B', 'n',
                                                 'I1', 't1', 'I2', 't2', 'Ipi', 'Ipc'])
        # 将文件夹名添加到文件路径
        outputFile = os.path.join(foldname, 'Current Waveform_CIGRE_S.xlsx')
        sheetName = f'Sheet{allp + 1}'
        if not os.path.exists(outputFile):
            mode = 'w'
        else:
            mode = 'a'

        with pd.ExcelWriter(outputFile, engine='openpyxl', mode=mode) as writer:
            df4.to_excel(writer, sheet_name=sheetName, index=False)
        filepath = foldname + '/' + 'Cigre Function_S.txt'
        # filepath = "H:\Ph.D. Python\StrongEMPPlatform\Tower\Predecessor\DATA_MCLG/b_Cigre Function.txt"
        # 打开文件并写入
        with open(filepath, 'w') as file:
            file.write(f"syms t\n")
            file.write(
                f"i(t)=((t<= tn).* (A*t+B*(t^n)) + (t>tn) * (I1*exp(-(t-tn)/t1) - I2*exp(-(t-tn)/t2)))*(Ipi/Ipc)\n")

    elif userInput4 == 1 and way == 2:
        df4 = pd.DataFrame(light_final, columns=['I0', 'nm', 't1', 'N', 't2'])
        # 将文件夹名添加到文件路径
        outputFile = os.path.join(foldname, 'Current Waveform_HEIDLER_S.xlsx')
        sheetName = f'Sheet{allp + 1}'
        if not os.path.exists(outputFile):
            mode = 'w'
        else:
            mode = 'a'

        with pd.ExcelWriter(outputFile, engine='openpyxl', mode=mode) as writer:
            df4.to_excel(writer, sheet_name=sheetName, index=False)
        filepath = foldname + '/' + 'Heidler Function_S.txt'
        # 打开文件并写入
        with open(filepath, 'w') as file:
            file.write(f"syms t\n")
            file.write(f"i(t)=(I0/nm)*(((t/t1)^N)/(1+(t/t1)^N))*exp(-t/t2)\n")

    if userInput4d == 1:  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        df4d = pd.DataFrame(np.array(light_final)[sited, :], columns=['tn', 'A', 'B', 'n',
                                                                      'I1', 't1', 'I2', 't2', 'Ipi', 'Ipc'])
        # 将文件夹名添加到文件路径
        outputFile = os.path.join(foldname, 'DIR Current Waveform CIGRE_S.xlsx')
        sheetName = f'Sheet{allp + 1}'
        if not os.path.exists(outputFile):
            mode = 'w'
        else:
            mode = 'a'

        with pd.ExcelWriter(outputFile, engine='openpyxl', mode=mode) as writer:
            df4d.to_excel(writer, sheet_name=sheetName, index=False)

    if userInput4ind == 1:  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        df4ind = pd.DataFrame(np.array(light_final)[siteind, :], columns=['tn', 'A', 'B', 'n',
                                                                          'I1', 't1', 'I2', 't2', 'Ipi', 'Ipc'])
        # 将文件夹名添加到文件路径
        outputFile = os.path.join(foldname, 'IND Current Waveform CIGRE_S.xlsx')
        sheetName = f'Sheet{allp + 1}'
        if not os.path.exists(outputFile):
            mode = 'w'
        else:
            mode = 'a'

        with pd.ExcelWriter(outputFile, engine='openpyxl', mode=mode) as writer:
            df4ind.to_excel(writer, sheet_name=sheetName, index=False)

    # 所有统计参数写成struct结构，用一个变量传递数据
    resultwave = {'light_final': light_final}

    return light_final
