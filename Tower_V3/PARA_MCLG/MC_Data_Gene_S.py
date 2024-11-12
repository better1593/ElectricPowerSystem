import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import re

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from matplotlib.path import Path
from multiprocessing import freeze_support
from shapely.geometry import LineString
from scipy.optimize import minimize_scalar
from scipy.optimize import differential_evolution
from scipy.spatial import distance
from sympy import symbols, exp, solve
from time import time

from Tower_V3.PARA_MCLG.lighting_parameters_distribution_S import lighting_parameters_distribution_S
from Tower_V3.PARA_MCLG.Lightning_stroke_location_S import Lightning_stroke_location_S
from Tower_V3.PARA_MCLG.current_waveform_generator_S import current_waveform_generator_S


np.seterr(divide="ignore", invalid="ignore")


class MC_Data_Gene_S():
    def __init__(self, path, FDIR):
        self.FDIR = FDIR
        self.path = path

    def MC_Data_Gene_S(self, GLB, Span, Tower, MC_lgtn, DSave,LatDis_max,Wave_Model):
        # init
        foldname = self.FDIR['dataMCLG2']
        NSpan = GLB['NSpan']
        NTower = GLB['NTower']
        averageh = MC_lgtn['averageh']

        # 假设只有SWnumber个shield wire和PCnumber个phase conductor,没有building
        # 初始化 Line 结构体的各个字段
        Line = {}
        # 初始化 Node 字段
        Line['Node_all'] = np.zeros((NTower, 5)) #matlab程序是(NTower,3)？？？？？
        # 初始化 Edge 字段
        Line['Edge_all'] = np.zeros((NSpan, 3))
        # 初始化 SWnumber 字段
        Line['SWnumber'] = np.zeros((1, NSpan), dtype=int)  # 每个span的shield wire的数量 取值范围[0,2]
        # 初始化 PCnumber 字段
        Line['PCnumber'] = np.zeros((1, NSpan), dtype=int)  # 每个span的phase conductor的数量 取值范围[1,10]
        # 初始化 segments 字段
        Line['segments'] = np.zeros((1, NSpan), dtype=int)
        # 初始化 Suppose_OHLP 字段
        Line['Suppose_OHLP'] = [None] * NSpan
        Line['Span'] = []
        Line['Tower'] = []

        for ik in range(NSpan):
            tid1 = Span[ik]['Info'][4]-1
            tid2 = Span[ik]['Info'][5]-1

            t1_coor = np.hstack((Tower[tid1]['Info'][4:7],[Tower[tid1]['Info'][10]]))
            t2_coor = np.hstack((Tower[tid2]['Info'][4:7],[Tower[tid2]['Info'][10]]))

            Line['Node_all'][tid1, 0:5] = np.hstack((tid1+1,t1_coor))
            Line['Node_all'][tid2, 0:5] = np.hstack((tid2+1,t2_coor))

            Cir = Span[ik]['Cir']
            Line['Edge_all'][ik, :3] = [ik+1, tid1+1, tid2+1]
            Line['SWnumber'][0, ik] = Cir['num'][0][1]
            Seg = Span[ik]['Seg']
            Line['segments'][0, ik] = Seg['Nseg']

            Cir['num']=Cir['num'].astype(int)
            tmp = np.zeros((Cir['num'][1][0] + 1, 2))
            cons = 1

            for jk in range(Cir['num'][0][1]):
                tmp[cons + jk, :2] = [Cir['dat'][jk][0], 0]
                cons += 1

            cirs = Cir['num'][0][1]

            for jk in range(Span[ik]['Cir']['num'][0][2]):
                tmp[cons+1-1, :2] = [Cir['dat'][cirs + jk][0], 1]
                tmp[cons+2-1, :2] = [Cir['dat'][cirs + jk][0], 2]
                tmp[cons+3-1, :2] = [Cir['dat'][cirs + jk][0], 3]
                cons += 3

            cirs += Cir['num'][0][2]

            for jk in range(Cir['num'][0][3]):
                tmp[cons + 1-1, :2] = [Cir['dat'][cirs + jk][0], 1]
                tmp[cons + 2-1, :2] = [Cir['dat'][cirs + jk][0], 2]
                tmp[cons + 3-1, :2] = [Cir['dat'][cirs + jk][0], 3]
                cons += 3

            cirs += Cir['num'][0][3]

            for jk in range(Cir['num'][0][4]):
                tmp[cons + 1-1, :2] = [Cir['dat'][cirs + jk][0], 1]
                tmp[cons + 2-1, :2] = [Cir['dat'][cirs + jk][0], 2]
                tmp[cons + 3-1, :2] = [Cir['dat'][cirs + jk][0], 3]
                tmp[cons + 4-1, :2] = [Cir['dat'][cirs + jk][0], 4]
                cons += 4

            tmpOHLP = np.vstack(
                (np.hstack((Span[ik]['Pole'], 0)),
                                Span[ik]['OHLP'][:, 0:7]))
            Line['Suppose_OHLP'][ik] = np.hstack((tmpOHLP, tmp))


            Line['Span'].append(Span[ik]['ID'])
            while len(Line['Tower']) <= tid1:
                Line['Tower'].append(0)
            Line['Tower'][tid1] = Tower[tid1]['ID']
            while len(Line['Tower']) <= tid2:
                Line['Tower'].append(0)
            Line['Tower'][tid2] = Tower[tid2]['ID']

        Itmp = np.where(Line['Node_all'][:, 0] == 0)[0]
        Line['Node_all'] = np.delete(Line['Node_all'], Itmp, axis=0)
        Line['Node_all'] = Line['Node_all'][Line['Node_all'][:, 0].argsort()]

        PoleXY = Line['Node_all']
        Npole = PoleXY.shape[0]

        TowerSW = np.zeros(Npole)
        for ik in range(Npole):
            hspn = Tower[ik]['T2Smap']['hspn']
            hsid = Tower[ik]['T2Smap']['hsid']
            tsid = Tower[ik]['T2Smap']['tsid']
            tspn = Tower[ik]['T2Smap']['tspn']

            spn = hspn + tspn
            sid = np.hstack((hsid, tsid))
            for jk in range(spn):
                jk_id = sid[jk]
                swno = Span[int(jk_id)-1]['Cir']['num'][0][1]
                if swno != 0:
                    TowerSW[ik] = 1
                    break

        xpoints = Line['Node_all'][:, 1:3]
        if any(xpoints[:, 1] != 0):
            raise ValueError('The position of pole-pole are \
            not matched the requirements of a straight line.')

        swheigh = Line['Suppose_OHLP'][0][1, 2]
        pnall = MC_lgtn['fixn'] * 200

        MC_lgtn['xpoints'] = xpoints
        MC_lgtn['swheight'] = swheigh

        sorted_coords = np.sort(xpoints[:, 0])
        mid_index = len(sorted_coords) // 2
        MC_lgtn['middlePoints'] = sorted_coords[mid_index-1:mid_index + 2-1]

        hp = LatDis_max / averageh
        MC_lgtn['pn'] = round(pnall / hp)

        Line['Suppose_OHLP2'] = [item[0:2, :] for item in Line['Suppose_OHLP']]

        # # 调用初始化函数
        # Line = self.Line_Init()
        # # struct获取每个变量名
        # Suppose_OHLP = Line.Suppose_OHLP
        # SWnumber = Line.SWnumber
        #
        # # 增强多样性, 第1条边没有sw，第2条边2根sw，第3条边pc有1相
        # Suppose_test = Suppose_OHLP[0, 0]
        # Suppose_test = np.delete(Suppose_test, 1, axis=0)
        # Suppose_OHLP[0, 0] = Suppose_test
        # SWnumber[0, 0] = 0
        # Suppose_test = Suppose_OHLP[1, 0]
        # Suppose_test3 = np.zeros((Suppose_test.shape[0] + 1, Suppose_test.shape[1]))
        # Suppose_test3[0: 2, :] = Suppose_test[0: 2, :]
        # Suppose_test3[2, :] = Suppose_test[1, :]
        # Suppose_test3[2, 0: 2] = Suppose_test[1, 0: 2] + 10
        # Suppose_test3[2, 3: 5] = Suppose_test[1, 3: 5] + 10
        # Suppose_test3[2, 7] = 2001
        # Suppose_test3[3, :] = Suppose_test[2, :]
        # Suppose_test3[4, :] = Suppose_test[3, :]
        # Suppose_test3[5, :] = Suppose_test[4, :]
        # Suppose_OHLP[1, 0] = Suppose_test3
        # SWnumber[0, 1] = 2
        # Suppose_test = Suppose_OHLP[2, 0]
        # Suppose_test[2, 1] = 0.5
        # Suppose_test[2, 4] = 0.5
        # Suppose_test[2, 6] = -0.5
        # Suppose_OHLP[2, 0] = Suppose_test

        # pole位置坐标
        # polexyall2 = []
        # for pw in range(len(Line['Suppose_OHLP'])):
        #     Suppose_OHLPe = Line['Suppose_OHLP'][pw]
        #     polexyall2.append(Suppose_OHLPe[0, :6])
        #
        # polexyall = np.array(polexyall2)
        # 将double 数组转换为table
        df92 = pd.DataFrame(PoleXY, columns=['Tower_ID','x','y','z','Height'])
        # 将文件夹名添加到文件路径
        outputFile = os.path.join(foldname, 'Pole_XY_S.xlsx')
        # 检查文件是否存在并删除
        try:
            os.remove(outputFile)  # 如果文件存在，删除它
        except FileNotFoundError:
            pass

        # 将 DataFrame 保存为 Excel 文件
        df92.to_excel(outputFile, index=False)

        # 函数
        Line['Node'] = Line['Node_all'][:, 1: 3]
        Line['Edges'] = Line['Edge_all'][:, 1: 3]

        for allp in range(int(hp)):
        # number of flashes and number of strokes
            [resultcur, parameterst,flash_stroke,points_need,XY_need3] = lighting_parameters_distribution_S (
                                                                                           allp,MC_lgtn,DSave,Line,foldname)

            # 判断落雷点位置
            [resultstro, dataSTS, stroke_result] = Lightning_stroke_location_S (allp,Line,resultcur,DSave,foldname)

            # 描述电流的波形
            light_final = current_waveform_generator_S (allp,Wave_Model,DSave,resultstro,resultcur,foldname)


        return XY_need3,parameterst,dataSTS,stroke_result,light_final,flash_stroke,PoleXY

