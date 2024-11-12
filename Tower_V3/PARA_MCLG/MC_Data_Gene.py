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

from Tower_V3.PARA_MCLG.pcnumber import pcnumber
from Tower_V3.PARA_MCLG.edge_image import edge_image
from Tower_V3.PARA_MCLG.lighting_parameters_distribution import lighting_parameters_distribution
from Tower_V3.PARA_MCLG.Lightning_stroke_location import Lightning_stroke_location
from Tower_V3.PARA_MCLG.current_waveform_generator import current_waveform_generator


np.seterr(divide="ignore", invalid="ignore")


class MC_Data_Gene():
    def __init__(self, path, FDIR):
        self.FDIR = FDIR
        self.path = path

    def MC_Data_Gene(self, GLB, Span, Tower, MC_lgtn, DSave,LatDis_max,Wave_Model,AR):
        # init
        foldname = self.FDIR['dataMCLG']
        NSpan = GLB['NSpan']
        NTower = GLB['NTower']

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


        # 输入斜坡的位置和坡度和高度；building左下角的位置和长，宽，高！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        Line['Slope_angle'] = np.array([[10,20],[20,30],[30,40]])  # 每条span的斜坡角度[上（左），下（右）各一个角度（度数）]，斜坡起点是最初的落雷点位置到最近的pole-pole线的垂足点
        #多个building
        Line['buildings'] = [
            {
                'building_XY': np.array([[50, 0], [70, 0], [50, 20], [70, 20]]),  # building四个点的XY坐标（顺序依次是左下角，右下角，左上角，右上角）
                'building_height': 8.5  # building的高度
            },
            {
                'building_XY': np.array([[30, 10], [40, 10], [30, 25], [40, 25]]),  # 第二个建筑物
                'building_height': 10.0
            }
        ]

        # 将double 数组转换为table
        df92 = pd.DataFrame(PoleXY, columns=['Tower_ID','x','y','z','Height'])
        # 将文件夹名添加到文件路径
        outputFile = os.path.join(foldname, 'Pole_XY.xlsx')
        # 检查文件是否存在并删除
        try:
            os.remove(outputFile)  # 如果文件存在，删除它
        except FileNotFoundError:
            pass


        # 将 DataFrame 保存为 Excel 文件
        df92.to_excel(outputFile, index=False)

        Line['OHLPf'] = pcnumber(Line)  # 判断phase conductor有几相-OHLPf

        # 函数
        Line['Node'] = Line['Node_all'][:, 1: 3]
        Line['Edges'] = Line['Edge_all'][:, 1: 3]

        Line['Suppose_OHLP'][0][1,2] = 9
        Line['Suppose_OHLP'][0][1, 5] = 9
        Line['Suppose_OHLP'][1][1, 2] = 9
        Line['Suppose_OHLP'][1][1, 5] = 9
        Line['Suppose_OHLP'][2][1, 2] = 9
        Line['Suppose_OHLP'][2][1, 5] = 9




        # 给定节点的坐标，边的连接情况和包围线的距离确定包围线
        [resultedge, XY_need3] = edge_image(Line, DSave, LatDis_max, foldname)

        # number of flashes and number of strokes
        [resultcur, parameterst, flash_stroke, points_need] = lighting_parameters_distribution (MC_lgtn, DSave,Line,
                                                                                               resultedge,foldname)

        # 判断落雷点位置
        [resultstro, dataSTS, stroke_result] = Lightning_stroke_location(Line, resultcur, DSave, foldname,
                                                                              resultedge,AR)

        # 描述电流的波形
        light_final = current_waveform_generator (Wave_Model, DSave, resultstro, resultcur, foldname)

        # 关于输出添加一列[flash id, stroke number, 直接（1）或间接（0）]
        DIND = stroke_result[0]
        DINDs = np.array([1 if x == 'Direct' else 0 for x in DIND])
        DIND2 = DINDs[resultcur['siteone']]
        flash_stroke = np.hstack((flash_stroke, DIND2.reshape(-1, 1)))
        # 将double 数组转换为table
        df27 = pd.DataFrame(flash_stroke, columns=['flash', 'stroke', 'Direct1_Indirect2'])
        # 将文件夹名添加到文件路径
        outputFile = os.path.join(foldname, 'Flash_Stroke Dist.xlsx')
        # 检查文件是否存在并删除
        try:
            os.remove(outputFile)  # 如果文件存在，删除它
        except FileNotFoundError:
            pass

        # 将 DataFrame 保存为 Excel 文件
        df27.to_excel(outputFile, index=False)

        return XY_need3,parameterst,dataSTS,stroke_result,light_final,flash_stroke,PoleXY

