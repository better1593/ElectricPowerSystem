## Edge_Image
# 结合纯数学的方法一和点不能在移动后的线段的四边形内的方法二得到XY_need(轮廓线的所有转折点)
# 具体解释见edge_test1和edge_test2程序
# 用图像处理的方法可以粗略的得到轮廓线，但是无法在轮廓线上标注转折点，因为它们的坐标系完全不一样
# 转折点是数学的xy坐标，轮廓线是像素坐标，两个坐标系无法对等变换
# 所以轮廓线的方法采取判断转折点XY_need是否在移动后的线段上决定哪个点连接哪个点

## Pcnumber
# 判断phase conductor有几相-OHLPf
# 先看dh正负，首先对于正的dh，找高度最高的，再找高度最高的中dh正的最大
# 对于负的dh，找高度最高的，再找高度最高的中dh负的最大
# 如果dh有正有负，则有2相；如果dh全是正/负，则有1相

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

np.seterr(divide="ignore",invalid="ignore")

class MC_Data_Generate():
    def __init__(self,path, FDIR):
        self.FDIR = FDIR
        self.path = path

    def MC_Data_Generate(self, GLB, Span):
        # init
        foldname = self.FDIR['MC_lightning_data']
        MC_lgtn = GLB['MC_lgtn']
        DSave = GLB['DSave']
        LatDis_max = GLB['LatDis_max']
        Wave_Model = GLB['Wave_Model']
        NSpan = GLB['NSpan']
        NTower = GLB['NTower']

        # 假设只有SWnumber个shield wire和PCnumber个phase conductor,没有building
        # 初始化 Line 结构体的各个字段
        Line = {}
        # 初始化 Node 字段
        Line['Node'] = np.zeros((NTower, 2))
        # 初始化 Edge 字段
        Line['Edge'] = np.zeros((NSpan, 2))
        # 初始化 SWnumber 字段
        Line['SWnumber'] = np.zeros((1, NSpan), dtype=int)   # 每个span的shield wire的数量 取值范围[0,2]
        # 初始化 PCnumber 字段
        Line['PCnumber'] = np.zeros((1, NSpan), dtype=int)    # 每个span的phase conductor的数量 取值范围[1,10]
        # 初始化 segments 字段
        Line['segments'] = np.zeros((1, NSpan), dtype=int)
        # 初始化 Suppose_OHLP 字段
        Line['Suppose_OHLP'] = [None] * NSpan

        for ik in range(0, NSpan):
            tid1 = Span[ik]['Info'][4]  # HTower id, Info = cell array
            tid2 = Span[ik]['Info'][5]  # TTower id, Info = cell array

            # Line.Span(ik)=Span(ik).ID;
            # Line.Tower(tid1)=Tower(tid1).ID;
            # Line.Tower(tid2)=Tower(tid2).ID;
            Line['Node'][tid1 - 1, 0:2] = Span[ik]['Pole'][0:2]
            Line['Node'][tid2 - 1, 0:2] = Span[ik]['Pole'][3:5]

            Seg = Span[ik]['Seg']
            Line['Edge'][ik, 0:2] = [tid1, tid2]
            Line['SWnumber'][0, ik] = Seg['Ngdw']  # SW#
            Line['segments'][0, ik] = Seg['Nseg']  # total # of segment
            Line['PCnumber'][0, ik] = Seg['Npha']  # phase cond #

            Cir = Span[ik]['Cir']
            Cir['num'] = Cir['num'].astype(int)

            tmp = np.zeros((Cir['num'][1, 0] + 1, 2))  # add a head line
            cons = 1
            for jk in range(1, Cir['num'][0, 1] + 1):  # SW
                tmp[cons + jk - 1, 0:2] = [Cir['dat'][jk - 1, 0], 1]
                cons = cons + 1
            cirs = Cir['num'][0, 1]
            for jk in range(1, Span[ik]['Cir']['num'][0, 2] + 1):  # 34.5kv
                tmp[cons + 0, 0:2] = [Cir['dat'][cirs, 0], 1]
                tmp[cons + 1, 0:2] = [Cir['dat'][cirs, 0], 2]
                tmp[cons + 2, 0:2] = [Cir['dat'][cirs, 0], 3]
                cons = cons + 3
                cirs = cirs + 1
            for jk in range(1, Cir['num'][0, 3] + 1):  # 10kV
                tmp[cons + 0, 0:2] = [Cir['dat'][cirs, 0], 1]
                tmp[cons + 1, 0:2] = [Cir['dat'][cirs, 0], 2]
                tmp[cons + 2, 0:2] = [Cir['dat'][cirs, 0], 3]
                cons = cons + 3
                cirs = cirs + 1
            for jk in range(1, Cir['num'][0, 4] + 1):  # 0.4kV
                tmp[cons + 0, 0:2] = [Cir['dat'][cirs, 0], 1]
                tmp[cons + 1, 0:2] = [Cir['dat'][cirs, 0], 2]
                tmp[cons + 2, 0:2] = [Cir['dat'][cirs, 0], 3]
                tmp[cons + 3, 0:2] = [Cir['dat'][cirs, 0], 4]
                cons = cons + 4
                cirs = cirs + 1
            OHLP = Span[ik]['OHLP'][:, 0:7]
            Itmp = np.append(Span[ik]['Pole'], 0)
            OHLP = np.vstack((Itmp, OHLP))

            Line['Suppose_OHLP'][ik] = np.concatenate((OHLP, tmp), axis=1)
        # pole位置坐标
        polexyall2 = []
        for pw in range(len(Line['Suppose_OHLP'])):
            Suppose_OHLPe = Line['Suppose_OHLP'][pw]
            polexyall2.append(Suppose_OHLPe[0, :6])

        polexyall = np.array(polexyall2)
        # 将double 数组转换为table
        df92 = pd.DataFrame(polexyall, columns=['x1','y1','z1','x2','y2','z2'])
        # 将文件夹名添加到文件路径
        outputFile = os.path.join(foldname, 'Tower_x_y_z.xlsx')
        # 检查文件是否存在并删除
        try:
            os.remove(outputFile)  # 如果文件存在，删除它
        except FileNotFoundError:
            pass

        # 将 DataFrame 保存为 Excel 文件
        df92.to_excel(outputFile, index=False)
        Line['OHLPf'] = self.Pcnumber(Line)  # 判断phase conductor有几相-OHLPf
        # 函数
        #给定节点的坐标，边的连接情况和包围线的距离确定包围线
        [resultedge, XY_need3] = self.Edge_Image(Line, DSave, LatDis_max, foldname)

        # number of flashes and number of strokes
        [resultcur,parameterst,flash_stroke,points_need] = self.Lighting_Parameters_Distribution(MC_lgtn, DSave, Line, resultedge, foldname)

        #判断落雷点位置
        [resultstro,dataSTS,stroke_result] = self.Lightning_Stroke_Location(Line, resultcur, DSave, foldname, resultedge)

        # 描述电流的波形
        resultwave = self.Current_Waveform_Generator(Wave_Model, DSave, resultstro, resultcur, foldname)

        # 关于输出添加一列[flash id, stroke number, 直接（1）或间接（2）]
        DIND = stroke_result[0]
        DINDs = np.array([1 if x == 'Direct' else 2 for x in DIND])
        DIND2 = DINDs[resultcur['siteone']]
        flash_stroke = np.hstack((flash_stroke, DIND2.reshape(-1, 1)))
        # 将double 数组转换为table
        df27 = pd.DataFrame(flash_stroke, columns=['flash', 'stroke','Direct1_Indirect2'])
        # 将文件夹名添加到文件路径
        outputFile = os.path.join(foldname, 'Flash_Stroke Dist.xlsx')
        # 检查文件是否存在并删除
        try:
            os.remove(outputFile)  # 如果文件存在，删除它
        except FileNotFoundError:
            pass

        # 将 DataFrame 保存为 Excel 文件
        df27.to_excel(outputFile, index=False)

        return resultedge,XY_need3,resultcur,parameterst,points_need,resultstro,dataSTS,stroke_result,resultwave,polexyall,flash_stroke

    # def Line_Init(self):
    #     # 假设只有SWnumber个shield wire和PCnumber个phase conductor, 没有building
    #     Node_all = np.array([
    #         [1, 0, 0],
    #         [2, 900, 0],
    #         [3, 1020, 0],
    #         [4, 1200, 0],
    #         [5, 1020, -150]
    #     ])
    #
    #     Edge_all = np.array([
    #         [1, 1, 2],
    #         [2, 2, 3],
    #         [3, 3, 4],
    #         [4, 3, 5]
    #     ])
    #
    #     SWnumber = np.array([  # 每个span的shield wire的数量 取值范围[0,2]
    #         [1, 1, 1, 1]
    #     ])
    #
    #     segments = np.array([
    #         [30, 4, 6, 5]
    #     ])
    #
    #     Suppose_OHLP = np.zeros([4, 1], dtype=object)  # 创建一个 4 行 1 列的NumPy 二维数组
    #     # 为每个元素赋值
    #     # 为每个元素赋值
    #     Suppose_OHLP[0, 0] = np.array([
    #         [0, 0, 10.5, 900, 0, 10.5, 0, 0, 0],
    #         [0, 0, 10.5, 900, 0, 10.5, 0, 1001, 0],
    #         [0, -0.5, 10, 900, -0.5, 10, 0.5, 3001, 1],
    #         [0, 0.1, 10, 900, 0.1, 10, -0.1, 3001, 2],
    #         [0, 0.6, 10, 900, 0.6, 10, -0.6, 3001, 3]
    #     ])
    #
    #     Suppose_OHLP[1, 0] = np.array([
    #         [900, 0, 10.5, 1020, 0, 10.5, 0, 0, 0],
    #         [900, 0, 10.5, 1020, 0, 10.5, 0, 1001, 0],
    #         [900, -0.5, 10, 1020, -0.5, 10, 0.5, 3001, 1],
    #         [900, 0.1, 10, 1020, 0.1, 10, -0.1, 3001, 2],
    #         [900, 0.6, 10, 1020, 0.6, 10, -0.6, 3001, 3]
    #     ])
    #
    #     Suppose_OHLP[2, 0] = np.array([
    #         [1020, 0, 10.5, 1200, 0, 10.5, 0, 0, 0],
    #         [1020, 0, 10.5, 1200, 0, 10.5, 0, 1001, 0],
    #         [1020, -0.5, 10, 1200, -0.5, 10, 0.5, 3001, 1],
    #         [1020, 0.1, 10, 1200, 0.1, 10, -0.1, 3001, 2],
    #         [1020, 0.6, 10, 1200, 0.6, 10, -0.6, 3001, 3]
    #     ])
    #
    #     Suppose_OHLP[3, 0] = np.array([
    #         [1020, 0, 10.5, 1020, -150, 10.5, 0, 0, 0],
    #         [1020, 0, 10.5, 1020, -150, 10.5, 0, 1001, 0],
    #         [1019.5, 0, 10, 1019.5, -150, 10, 0.5, 3001, 1],
    #         [1020.1, 0, 10, 1020.1, -150, 10, -0.1, 3001, 2],
    #         [1020.6, 0, 10, 1020.6, -150, 10, -0.6, 3001, 3]
    #     ])
    #
    #     Span = np.array([
    #         [1, 2, 3, 4]
    #     ])
    #
    #     Tower = np.array([
    #         [1, 2, 3, 4, 5]
    #     ])
    #
    #     return Node_all, Edge_all, SWnumber, segments, Suppose_OHLP, Span, Tower
    #
    #     # Line = LineConfig(Node_all, Edge_all, SWnumber, segments, Suppose_OHLP, Span, Tower)
    #     Line = Line_Initf(Node_all, Edge_all, SWnumber, segments, Suppose_OHLP, Span, Tower)
    #     return Line

    def Pcnumber(self, Line):
        # struct获取每个变量名
        # Suppose_OHLP = np.array(Line['Suppose_OHLP'])
        SWnumber = Line['SWnumber']
        Suppose_OHLP = Line['Suppose_OHLP']
        # OHLPf = np.zeros((Suppose_OHLP.shape[0], Suppose_OHLP.shape[0] + 2), dtype=object)
        # OHLPf = np.zeros((len(Suppose_OHLP), len(Suppose_OHLP[0]) + 2), dtype=object)
        # OHLPf = np.zeros((Suppose_OHLP.shape[0], 1 + 2), dtype=object)
        OHLPf = np.zeros((len(Suppose_OHLP), 1 + 3), dtype=object)

        for pw in range(len(Suppose_OHLP)):
            Suppose_OHLPe = Suppose_OHLP[pw]
            dh_all = np.zeros((Suppose_OHLPe.shape[0], 1), dtype=object)
            dh_all[(SWnumber[0, pw] + 1):, 0] = Suppose_OHLPe[(SWnumber[0, pw] + 1):, 6]
            Indices1 = np.where(dh_all > 0)[0] ## 和matlab不一样??????????????????????Indices3和Indices6不需要改变，我已经考虑了它们的变化，我需要的是位置牵引？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？
            if Indices1.size > 0:
                Indices12 = np.where(Suppose_OHLPe[Indices1, 2] == max(Suppose_OHLPe[Indices1, 2]))[0]
                Indices2 = Indices1[Indices12]
                Indices23 = np.where(Suppose_OHLPe[Indices2, 6] == max(Suppose_OHLPe[Indices2, 6]))[0]
                Indices3 = Indices2[Indices23]
            Indices4 = np.where(dh_all < 0)[0]
            if Indices4.size > 0:
                Indices45 = np.where(Suppose_OHLPe[Indices4, 2] == max(Suppose_OHLPe[Indices4, 2]))[0]
                Indices5 = Indices4[Indices45]
                Indices56 = np.where(Suppose_OHLPe[Indices5, 6] == min(Suppose_OHLPe[Indices5, 6]))[0]
                Indices6 = Indices5[Indices56]
            if 'Indices3' in locals() and 'Indices6' in locals():
                OHLPf[pw, 0] = Suppose_OHLPe[np.concatenate((np.arange(SWnumber[0, pw] + 1), Indices3, Indices6)), :]
                # OHLPf[pw, 1] = [Indices3  + 1, Indices6  + 1]
                OHLPf[pw, 1] = [Indices3, Indices6]
                OHLPf[pw, 2] = Suppose_OHLPe[:, 7:]
                OHLPf[pw, 3] = [Indices3, Indices6]
            elif 'Indices3' in locals() and 'Indices6' not in locals():
                OHLPf[pw, 0] = Suppose_OHLPe[np.concatenate((np.arange(SWnumber[0, pw] + 1), Indices3)), :]
                # OHLPf[pw, 1] = Indices3 + 1
                OHLPf[pw, 1] = Indices3
                OHLPf[pw, 2] = Suppose_OHLPe[:, 7:]
                OHLPf[pw, 3] = [Indices3]
            elif 'Indices3' not in locals() and 'Indices6' in locals():
                OHLPf[pw, 0] = Suppose_OHLPe[np.concatenate((np.arange(SWnumber[0, pw] + 1), Indices6)), :]
                # OHLPf[pw, 1] = Indices6 + 1
                OHLPf[pw, 1] = Indices6
                OHLPf[pw, 2] = Suppose_OHLPe[:, 7:]
                OHLPf[pw, 3] = [Indices6]
            else:
                OHLPf[pw, 0] = Suppose_OHLPe[:(SWnumber[0, pw] + 1), :]
                OHLPf[pw, 1] = np.nan
                OHLPf[pw, 2] = np.nan
                OHLPf[pw, 3] = np.nan

            # 删除多个变量
            try:
                del Indices3, Indices6
            except NameError:
                pass  # 如果变量不存在，忽略错误
        # OHLPf = list(list(OHLPf))
        return OHLPf

    def Edge_Image(self, Line, DSave, LatDis_max, foldname):
        # struct获取每个变量名
        Coordinates = Line['Node']
        Edges = Line['Edge'].astype(int)
        shift_vec = LatDis_max
        userInput1 = DSave['userInput1']

        # python特定排序从0开始
        Edges -= 1

        # 初始化一个点连接计数器
        count = np.zeros([Coordinates.shape[0], 1], dtype=object)

        # 遍历所有边
        for i in range(Edges.shape[0]):
            # 对于每条边，将起点和终点的计数器加1
            count[Edges[i, 0]] += 1
            count[Edges[i, 1]] += 1

        # 找出只连接了一个点的点
        idx1 = np.where(count == 1)[0]
        single_point_coordinates = Coordinates[idx1, :]

        # 将只连接了一个点的点按照它所在线段的方向向量的反方向移动1个单位长度
        p_new = np.zeros([single_point_coordinates.shape[0], 2], dtype=object)

        for i in range(single_point_coordinates.shape[0]):
            point_to_move = single_point_coordinates[i, :]
            idx_move = idx1[i]
            # 查找点point_to_move在Coordinates中的行数_idx1
            # 查找点p在Edges中的行数
            idx = np.where((Edges[:, 0] == idx_move) | (Edges[:, 1] == idx_move))[0]
            if Edges[idx, 0] == idx_move:
                e = Edges[idx, 1]
            else:
                e = Edges[idx, 0]

            direction = Coordinates[e, :] - point_to_move
            unit_vector = direction / np.linalg.norm(direction)
            p_new[i, :] = point_to_move - shift_vec * unit_vector

        # 把只连接了一个点的点替换为移动了1个单位长度的新点-p_new
        Coordinates_new = np.copy(Coordinates)
        for i in range(single_point_coordinates.shape[0]):
            Coordinates_new[idx1[i], :] = p_new[i, :]

        # 绘制第一张图
        plt.figure(1)
        for i in range(Edges.shape[0]):
            pt1 = Coordinates[Edges[i, 0], :]
            pt2 = Coordinates[Edges[i, 1], :]
            plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], linewidth=2)

        plt.axis('equal')
        plt.title('Original Polygon')
        plt.show()
        # 绘制第二张图
        plt.figure(2)
        for i in range(Edges.shape[0]):
            pt1 = Coordinates_new[Edges[i, 0], :]
            pt2 = Coordinates_new[Edges[i, 1], :]
            plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], linewidth=2)

        plt.axis('equal')
        plt.title('New Polygon')
        plt.show()
        # 初始化已有的线段列表
        exist_line = []
        lineidx = 0

        # 创建第3张图
        plt.figure(3)

        node_data = []
        for i in range(Edges.shape[0]):
            pt1 = Coordinates_new[Edges[i, 0], :]
            pt2 = Coordinates_new[Edges[i, 1], :]
            vec = pt2 - pt1
            unit_normal = [vec[1], -vec[0]] / np.linalg.norm(vec)

            pt1_shifted_pos = pt1 + shift_vec * unit_normal
            pt1_shifted_neg = pt1 - shift_vec * unit_normal
            pt2_shifted_pos = pt2 + shift_vec * unit_normal
            pt2_shifted_neg = pt2 - shift_vec * unit_normal

            # 每条线段的两端移动后产生的两个点的坐标:1是端点1的pos,2是端点1的neg,4是端点2的pos,3是端点2的neg
            # 注意顺序，顺序必须是1,2,3,4，才可以4个点依次连接形成多边形
            node_data.append([[pt1_shifted_pos[0], pt1_shifted_pos[1]],
                              [pt1_shifted_neg[0], pt1_shifted_neg[1]],
                              [pt2_shifted_neg[0], pt2_shifted_neg[1]],
                              [pt2_shifted_pos[0], pt2_shifted_pos[1]]])

            # 绘制移动后的2条线段
            plt.plot([pt1_shifted_pos[0], pt2_shifted_pos[0]], [pt1_shifted_pos[1], pt2_shifted_pos[1]], 'y',
                     linewidth=2)
            exist_line.append([[pt1_shifted_pos[0], pt1_shifted_pos[1]], [pt2_shifted_pos[0], pt2_shifted_pos[1]]])
            lineidx += 1
            plt.plot([pt1_shifted_neg[0], pt2_shifted_neg[0]], [pt1_shifted_neg[1], pt2_shifted_neg[1]], 'y',
                     linewidth=2)
            exist_line.append([[pt1_shifted_neg[0], pt1_shifted_neg[1]], [pt2_shifted_neg[0], pt2_shifted_neg[1]]])
            lineidx += 1
            # 连接孤立点未闭合的两端
            if np.any(np.all(p_new == pt1, axis=1)):
                plt.plot([pt1_shifted_pos[0], pt1_shifted_neg[0]], [pt1_shifted_pos[1], pt1_shifted_neg[1]], 'y',
                         linewidth=2)
                exist_line.append([[pt1_shifted_pos[0], pt1_shifted_pos[1]], [pt1_shifted_neg[0], pt1_shifted_neg[1]]])
                lineidx += 1

            if np.any(np.all(p_new == pt2, axis=1)):
                plt.plot([pt2_shifted_pos[0], pt2_shifted_neg[0]], [pt2_shifted_pos[1], pt2_shifted_neg[1]], 'y',
                         linewidth=2)
                exist_line.append([[pt2_shifted_pos[0], pt2_shifted_pos[1]], [pt2_shifted_neg[0], pt2_shifted_neg[1]]])
                lineidx += 1

            plt.axis('equal')

        node_data = np.array(node_data)  # list转换为np
        exist_line = np.array(exist_line)

        # 绘制已有线段
        for line in exist_line:
            plt.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], 'y', linewidth=2)

        # 绘图
        # plt.show()

        # 求解所有点包括所有端点和所有新产生的交点-[Xall',Yall']
        Xall = []
        Yall = []
        Xall_point = []  # 新产生的点的两条线段的四个交点的x坐标
        Yall_point = []  # %新产生的点的两条线段的四个交点的y坐标
        # 两条线段所求得交点
        for i in range(len(exist_line) - 1):
            X1, Y1 = exist_line[i][0]
            X2, Y2 = exist_line[i][1]

            for su in range(i + 1, len(exist_line)):
                X3, Y3 = exist_line[su][0]
                X4, Y4 = exist_line[su][1]

                # 求两线段交点的x,y坐标
                line1 = LineString([(X1, Y1), (X2, Y2)])
                line2 = LineString([(X3, Y3), (X4, Y4)])
                intersection = line1.intersection(line2)
                intspoint_x, intspoint_y = intersection.xy

                if len(intspoint_x) == 1:  # % 若两条线段无交点则跳至下一组线段，若有交点则将交点的x,y坐标存至S中!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    Xall.append(intspoint_x[0])
                    Yall.append(intspoint_y[0])
                    Xall_point.append([X1, X2, X3, X4])
                    Yall_point.append([Y1, Y2, Y3, Y4])
        plt.plot(Xall, Yall, 'c*')
        # plt.show()
        # 所有端点-node_data3
        node_data3 = []  # 列表
        for i in range(node_data.shape[1]):
            node_datae = np.concatenate(node_data[:, i], axis=0).reshape(-1, 2)
            node_data3.append(node_datae)

        node_data3 = np.vstack(node_data3)

        # 新产生的交点-node_generate
        Xall = np.array(Xall)
        Yall = np.array(Yall)
        XYall = np.array([Xall, Yall]).T
        # 将每行转为字符串以进行 setdiff1d 操作
        XYall_str = np.array([' '.join(map(str, row)) for row in XYall], dtype='str')
        node_data3_str = np.array([' '.join(map(str, row)) for row in node_data3], dtype='str')
        # 在扁平化的数组中找到差异
        node_generate_str = np.setdiff1d(XYall_str, node_data3_str, assume_unique=True)
        # 将结果转回为二维数组
        # 新产生的交点-node_generate
        node_generate = np.array([list(map(float, row.split())) for row in node_generate_str])
        ################## 轮廓线所有的转折点-XY_need
        XY_need = XYall
        # 结合移动后产生的线段的区间
        # 已求出所有的交点，轮廓线的点的本质是不在每条线段移动后形成的区间内
        # 所以删除在每条线段移动后形成的区间内的点就是轮廓线的点
        for i in range(node_data.shape[0]):
            # 每条线段的多边形的四个顶点的xy坐标
            edge_node = node_data[i, :][np.newaxis, :, :]
            # 初始化 x 和 y 列向量
            x_coords = np.zeros((edge_node.shape[0], edge_node.shape[1]))
            y_coords = np.zeros((edge_node.shape[0], edge_node.shape[1]))
            # 循环遍历 edge_node 中的每个元素，提取 x 坐标和 y 坐标
            for h in range(edge_node.shape[1]):
                x_coords[0, h] = edge_node[0, h][0]
                y_coords[0, h] = edge_node[0, h][1]

            # 四边形的四个顶点
            xv = x_coords.reshape(-1, 1)
            yv = y_coords.reshape(-1, 1)
            # 当多边形是封闭的，一定首尾相接
            xv = np.vstack((xv, xv[0]))
            yv = np.vstack((yv, yv[0]))
            # 待判断点的横坐标和纵坐标
            xq = XY_need[:, 0]
            yq = XY_need[:, 1]
            # 判断点是否在四边形内或线上
            # in 是在四边形内，on 是在四边形线上
            # 添加容错度的方法：将多边形的边界扩展一个小量
            tolerance = 1e-100  # 设置容错度
            xv_extended = xv.flatten() + tolerance
            yv_extended = yv.flatten() + tolerance
            # 创建 Path 对象
            polygon_path = Path(np.column_stack((xv_extended, yv_extended)))
            # 判断点是否在四边形内或线上
            inside_polygon = polygon_path.contains_points(np.column_stack((xq, yq)))
            on_polygon = np.isclose(np.array([polygon_path.contains_point((x, y)) for x, y in zip(xq, yq)]), True)
            # 考虑到计算机中使用的是有限精度的浮点数表示方法，因此在比较浮点数时会出现舍入误差
            # 导致出现错误判断两条线段的交点在四边形内
            # 如果判断点P到多边形线段AB的距离小于或等于1e-7，inpolygon函数将返回1，否则返回0
            con = np.zeros((xq.shape[0], 1))
            di = np.zeros(((xv.shape[0] - 1), 1))
            for wi in range(xq.shape[0]):
                for we in range(xv.shape[0] - 1):
                    xa, ya = xv[we, 0], yv[we, 0]
                    xb, yb = xv[we + 1, 0], yv[we + 1, 0]
                    xp, yp = xq[wi], yq[wi]
                    di[we, 0] = np.abs((xb - xa) * (ya - yp) - (xa - xp) * (yb - ya)) / np.sqrt(
                        (xb - xa) ** 2 + (yb - ya) ** 2)

                if any(di <= 1e-7):
                    con[wi, 0] = 1

            # 删除在四边形内且不在四边形线上的点(因为python和matlab程序本身的准确度不一样，所以python只要求in和con两方面，忽略on,结果与matlab一样)
            mask = (inside_polygon) & (con[:, 0] == 0)
            xq = xq[~mask]
            yq = yq[~mask]
            # 更新XY_need
            XY_need = np.column_stack((xq, yq))

        # 点的数值必须是准确的，不能是舍入误差后的值，因为后面需要判断ismember
        # 继续在第3张图,用红色标注轮廓线所有的转折点-XY_need
        plt.figure(3)
        plt.plot(XY_need[:, 0], XY_need[:, 1], 'r*')
        # 把包围线和原始图放在一起，形成更鲜明的对比
        # 绘制宽度=10的多边形
        plt.figure(3)
        for i in range(len(Edges)):
            pt1 = Coordinates[Edges[i, 0]]
            pt2 = Coordinates[Edges[i, 1]]
            plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'm-', linewidth=2)

        plt.title('Shifted Polygon and Original Polygon')
        plt.axis('equal')
        plt.show()
        #############确定连接的线段
        # 轮廓线所有的转折点-XY_need里面有重复的点，删除重复的点
        XY_need = np.unique(XY_need, axis=0)
        # 记录轮廓线的线段-edge_line
        edge_line = []
        edgeidx = 1
        # 对于 exist_line 的每一条线段
        for i in range(exist_line.shape[0]):
            D = []
            Didx = 1

            for gk in range(XY_need.shape[0]):
                # 判断是否有XY_need的其他点在线段上
                A = XY_need[gk, :]
                B = exist_line[i, 0, :]
                C = exist_line[i, 1, :]
                # 计算点1和点2之间的距离
                distance1 = distance.euclidean(B, C)
                distance2 = distance.euclidean(A, B)
                distance3 = distance.euclidean(A, C)
                # 计算点 A 是否在线段 BC 上
                # 一定要根据隔离线的距离决定保留小数点后几位，不能是两个距离直接等于
                # 因为当隔离线的距离是小数时, 可能两个距离可能相差很微小的值，所以要用round保留
                if round(distance1 - (distance2 + distance3), 4) == 0:
                    D.append(A)
                    Didx += 1

            D = np.array(D)
            # 有多少 XY_need 的其他点在线段上 - Didx
            Didx -= 1
            # 如果有偶数个XY_need的其他点（包括两个顶点）在线段上（≥2）, 比如
            # 如果只有一个顶点属于XY_need且有且只有一个XY_need的其他点在线段上
            # 如果两个顶点都不属于XY_need且有两个XY_need的其他点在线段上
            # 如果两个顶点都属于XY_need且没有XY_need的其他点在线段上
            # 则按照x的大小顺序，直接两两连接点
            if Didx >= 2 and Didx % 2 == 0:
                # 按照 x 的大小顺序
                sorted_D = D[np.argsort(D[:, 0])]
                for ls in range(0, Didx, 2):
                    E = sorted_D[ls, :]
                    F = sorted_D[ls + 1, :]
                    edge_line.append([E[0], E[1], F[0], F[1]])
                    edgeidx += 1
                    plt.plot([E[0], F[0]], [E[1], F[1]], 'k-', linewidth=2)

                continue

            # 随着隔离线距离的不同，XY_need可能会出现的错误
            # 如果两个顶点都属于XY_need且有一个XY_need的其他点在线段上（这种情况其实是因为有一个顶点被错误的判定为轮廓点XY_need）
            # 如果在线段上的XY_need的其他点已经与其中一个顶点连接了，取消它们的连接
            # 则连接两个顶点中已经跟其他点连接的那个顶点和在线段上的点
            # 这种情况需要最后判断, 所以先记录为XY_need_spe
            if Didx == 3:
                if np.all(np.any(np.all(exist_line[i, 0, :] == XY_need, axis=1))) and np.all(
                        np.any(np.all(exist_line[i, 1, :] == XY_need, axis=1))):
                    # 创建一个空的 NumPy 数组
                    XY_need_spe = np.empty((0, 2), dtype=float)
                    # XY_need_spe 第一行是在线段上的点 D
                    # 将每行转为字符串以进行 setdiff1d 操作
                    D_str = np.array([' '.join(map(str, row)) for row in D], dtype='str')
                    B2 = [B]  # 1维变2维
                    B_str = np.array([' '.join(map(str, row)) for row in B2], dtype='str')
                    # 在扁平化的数组中找到差异
                    DB_str = np.setdiff1d(D_str, B_str, assume_unique=True)
                    # 将结果转回为二维数组
                    # setdiff(D,B,'rows')
                    DB = np.array([list(map(float, row.split())) for row in DB_str])
                    # 将每行转为字符串以进行 setdiff1d 操作
                    DB_str = np.array([' '.join(map(str, row)) for row in DB], dtype='str')
                    C2 = [C]  # 1维变2维
                    C_str = np.array([' '.join(map(str, row)) for row in C2], dtype='str')
                    # 在扁平化的数组中找到差异
                    DBC_str = np.setdiff1d(DB_str, C_str, assume_unique=True)
                    # 将结果转回为二维数组
                    # setdiff(setdiff(D,B,'rows'),C,'rows')
                    DBC = np.array([list(map(float, row.split())) for row in DBC_str])
                    # 添加第一行数据
                    XY_need_spe = np.append(XY_need_spe, DBC, axis=0)
                    # XY_need_spe 第二行和第三行是已有线段的两个顶点 B 和 C
                    XY_need_spe = np.append(XY_need_spe, [B], axis=0)
                    XY_need_spe = np.append(XY_need_spe, [C], axis=0)

                continue
        # plt.show()
        # 记录轮廓线的线段 edge_line 里面有重复的线段，删除重复的线段生成 edge_line2
        edge_lineall = np.array(edge_line)
        edge_line2 = np.unique(edge_lineall, axis=0)
        # 如果某条线段两个顶点是一个点，则删除这条线段，生成 edge_line2
        edge_line0 = edge_line2.copy()
        for i in range(edge_line2.shape[0]):
            if np.all(edge_line2[i, :2] == edge_line2[i, 2:]):
                edge_line0[i, :] = [0, 0, 0, 0]

        edge_line2 = edge_line0[edge_line0[:, 0] != 0]
        # 随着包围线距离的不同，XY_need 可能会错误判定某一个点
        # 如果有错误，则会产生 XY_need_spe
        if 'XY_need_spe' in locals():
            # XY_need_spe，如果在线段上的 XY_need 的其他点已经与其中一个顶点连接了，取消它们的连接
            # 在线段上的点D与已有线段的一个顶点B
            line_delete = np.concatenate([XY_need_spe[0, :], XY_need_spe[1, :]])
            # 在线段上的点D与已有线段的一个顶点C
            line_delete2 = np.concatenate([XY_need_spe[0, :], XY_need_spe[2, :]])
            # 取消它们的连接
            # edge_line3=setdiff(edge_line2,line_delete, 'rows');
            # 将每行转为字符串以进行 setdiff1d 操作
            edge_line2_str = np.array([' '.join(map(str, row)) for row in edge_line2], dtype='str')
            line_delete_str = np.array([' '.join(map(str, row)) for row in [line_delete]], dtype='str')
            # 在扁平化的数组中找到差异
            edge_line3_str = np.setdiff1d(edge_line2_str, line_delete_str, assume_unique=True)
            # 将结果转回为二维数组
            edge_line3 = np.array([list(map(float, row.split())) for row in edge_line3_str])
            # edge_line4=setdiff(edge_line3,line_delete2, 'rows');
            # 将每行转为字符串以进行 setdiff1d 操作
            edge_line3_str = np.array([' '.join(map(str, row)) for row in edge_line3], dtype='str')
            line_delete2_str = np.array([' '.join(map(str, row)) for row in [line_delete2]], dtype='str')
            # 在扁平化的数组中找到差异
            edge_line4_str = np.setdiff1d(edge_line3_str, line_delete2_str, assume_unique=True)
            # 将结果转回为二维数组
            edge_line4 = np.array([list(map(float, row.split())) for row in edge_line4_str])
            # 则连接两个顶点中已经跟其他点连接的那个顶点和在线段上的点
            edge_line4_start = edge_line4[:, :2]
            edge_line4_end = edge_line4[:, 2:]
            if np.all(np.any(np.all(XY_need_spe[1, :] == edge_line4_start, axis=1))) or np.all(
                    np.any(XY_need_spe[1, :] == edge_line4_end, axis=1)):
                # 轮廓线上的转折点 XY_need2 (删除了 XY_need 里错误的点）
                # XY_need2=setdiff(XY_need,XY_need_spe(3,:),'rows');
                spe_delete = XY_need_spe[2, :]
                XY_need_str = np.array([' '.join(map(str, row)) for row in XY_need], dtype='str')
                spe_delete_str = np.array([' '.join(map(str, row)) for row in [spe_delete]], dtype='str')
                XY_need2_str = np.setdiff1d(XY_need_str, spe_delete_str, assume_unique=True)
                XY_need2 = np.array([list(map(float, row.split())) for row in XY_need2_str])
                new_el = np.array([np.array([XY_need_spe[1, :], XY_need_spe[0, :]]).flatten()])
                edge_line4 = np.concatenate([edge_line4, new_el])
            else:
                # XY_need2=setdiff(XY_need,XY_need_spe(2,:),'rows');
                spe_delete2 = XY_need_spe[1, :]
                XY_need_str = np.array([' '.join(map(str, row)) for row in XY_need], dtype='str')
                spe_delete2_str = np.array([' '.join(map(str, row)) for row in [spe_delete2]], dtype='str')
                XY_need2_str = np.setdiff1d(XY_need_str, spe_delete2_str, assume_unique=True)
                XY_need2 = np.array([list(map(float, row.split())) for row in XY_need2_str])
                new_el = np.array([np.array([XY_need_spe[2, :], XY_need_spe[0, :]]).flatten()])
                edge_line4 = np.concatenate([edge_line4, new_el])

            # 无论 XY_need 是否有错误的点，保持不同的包围线距离的轮廓线和转折点的变量命名都是一样的
            edge_line2 = edge_line4

        # 如果两个顶点都属于 XY_need 且有一个 XY_need 的其他点在线段上
        # （这种情况其实是因为有一个顶点被错误的判定为轮廓点XY_need）
        # 有一些点不满足这个情况所以没有被检测到是错误的点而没有被删除
        # 最不出错的方式是轮廓线 edge_line2 的点是转折点 XY_need
        XY_need3 = []
        for j in range(edge_line2.shape[0]):
            XY_need3.append([edge_line2[j, 0], edge_line2[j, 1]])
            XY_need3.append([edge_line2[j, 2], edge_line2[j, 3]])

        XY_need3 = np.array(XY_need3)
        XY_need4 = np.unique(XY_need3, axis=0)
        XY_need = XY_need4
        ################ 生成轮廓线的最终结果
        # 轮廓线（哪个点连接哪个点）-edge_line2（1,2列是线段的一个顶点；3,4列是线段的另一个顶点）
        # 轮廓线上的转折点XY_need
        # 画图-标注转折点和轮廓线
        plt.figure(4)
        plt.plot(XY_need[:, 0], XY_need[:, 1], 'r*')
        for i in range(edge_line2.shape[0]):
            plt.plot([edge_line2[i, 0], edge_line2[i, 2]], [edge_line2[i, 1], edge_line2[i, 3]], 'k-', linewidth=2)
        # 把包围线和原始图放在一起，形成更鲜明的对比
        # 绘制宽度=10的多边形
        for i in range(Edges.shape[0]):
            pt1 = Coordinates[Edges[i, 0], :]
            pt2 = Coordinates[Edges[i, 1], :]
            plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'g-', linewidth=2)
        # 设置坐标轴等参数
        plt.axis('equal')
        # plt.show()
        # 当出现两个转折点没有连接时，需要延长固定距离的隔离线直至相交
        edge_line2_A = edge_line2[:, :2]
        edge_line2_B = edge_line2[:, 2:]
        edge_line2_AB = np.vstack((edge_line2_A, edge_line2_B))
        # 计算每个转折点连接了几个点，找到孤立点-solopoints
        edge_line2_ABu, icAB = np.unique(edge_line2_AB, axis=0, return_inverse=True)
        countsAB = np.bincount(icAB)
        resultAB = np.column_stack((edge_line2_ABu, countsAB))
        solopoints = resultAB[resultAB[:, 2] == 1, :2]
        # 对孤立点配对，连接离它距离最近的孤立点-idxpair
        # 对孤立点配对，不仅距离是最近的，而且是在被画包围线的多边形的同一侧
        # 计算每个点到其他所有点的距离
        n = solopoints.shape[0]
        distances = np.zeros((n, n))
        # 当包围线距离很小时，会导致正确配对的两个点的直接连线与多边形的某条边相交
        # 改成新判断条件：对孤立点配对，不仅距离是最近的，而且不是同一条边移动后的轮廓线相交形成的交点
        pa1 = (-1) * np.ones((n, 1))
        pa2 = (-1) * np.ones((n, 1))
        for i in range(n):
            X1, Y1 = solopoints[i, 0], solopoints[i, 1]
            psidx = np.all(XYall == [X1, Y1], axis=1)
            ps = (np.where(psidx)[0])
            Xall_point = np.array(Xall_point)
            Yall_point = np.array(Yall_point)
            point1 = [Xall_point[ps, 0], Yall_point[ps, 0]]
            point2 = [Xall_point[ps, 1], Yall_point[ps, 1]]  # 一条线段的两个端点
            point1 = np.array(point1).T
            point2 = np.array(point2).T
            for zm in range(node_data.shape[0]):
                if ((np.all(node_data[zm, 0] == point1) and np.all(node_data[zm, 3] == point2)) or
                        (np.all(node_data[zm, 3] == point1) and np.all(node_data[zm, 0] == point2))):
                    pa1[i] = zm

                if ((np.all(node_data[zm, 1] == point1) and np.all(node_data[zm, 2] == point2)) or
                        (np.all(node_data[zm, 2] == point1) and np.all(node_data[zm, 1] == point2))):
                    pa1[i] = zm

            point3 = [Xall_point[ps[0], 2], Yall_point[ps[0], 2]]
            point4 = [Xall_point[ps[0], 3], Yall_point[ps[0], 3]]  # 另一条线段的两个端点
            point3 = np.array(point3)
            point4 = np.array(point4)
            for zm2 in range(node_data.shape[0]):
                if ((np.all(node_data[zm2, 0] == point3) and np.all(node_data[zm2, 3] == point4)) or
                        (np.all(node_data[zm2, 3] == point3) and np.all(node_data[zm2, 0] == point4))):
                    pa2[i] = zm2

                if ((np.all(node_data[zm2, 1] == point3) and np.all(node_data[zm2, 2] == point4)) or
                        (np.all(node_data[zm2, 2] == point3) and np.all(node_data[zm2, 1] == point4))):
                    pa2[i] = zm2

        # 矩阵 distances 的每一行是一个点到其他所有点的距离
        for i in range(n):
            for j in range(n):
                distances[i, j] = np.linalg.norm(solopoints[i, :] - solopoints[j, :])

        # 对角线是点到它本身的距离，设为最大，不影响找距离最小的点
        distances = distances + np.diag(np.full((n,), np.inf))
        distances2 = distances.copy()
        idxpair = np.zeros((n, 2), dtype=int)

        for i in range(n):
            found = False  # 初始化标志，表示是否找到符合条件的配对的孤立点
            while not found:
                distanceidx = np.where(distances[i, :] == np.min(distances[i, :]))[0]

                # 如果找不到有两个孤立点不是同一条边移动后的轮廓线相交形成的交点，直接选择距离最近的点
                if len(distanceidx) > 1:
                    distanceidx2 = np.where(distances2[i, :] == np.min(distances2[i, :]))[0]
                    found = True  # 找到符合条件的配对的孤立点
                    idxpair[i, :] = [i, distanceidx2[0]]
                else:
                    # 若配对的孤立点不是同一条边移动后的轮廓线相交形成的交点
                    if pa1[i] != pa1[distanceidx] and pa1[i] != pa2[distanceidx] \
                            and pa2[i] != pa1[distanceidx] and pa2[i] != pa2[distanceidx]:
                        found = True  # 找到符合条件的配对的孤立点
                        idxpair[i, :] = [i, distanceidx[0]]
                    else:
                        distances[i, distanceidx] = np.inf

        # 删除idxpair距离近且又不是同一条边移动后的轮廓线相交形成的交点的错误的孤立点配对
        idxpair = np.sort(idxpair, axis=1)
        # 计算每行出现次数
        uniqueRows, ic = np.unique(idxpair, axis=0, return_inverse=True)
        counts = np.bincount(ic)
        # 找出出现次数为 2 的行-配对正确的孤立点对
        nosingleOccurrences = uniqueRows[counts == 2, :]
        # 初始化一个逻辑索引，用于标记要保留的行
        keepRows = np.ones((idxpair.shape[0],), dtype=bool)
        # 遍历出现次数为 2 的行
        for i in range(nosingleOccurrences.shape[0]):
            # 检查每个元素是否在其他行中出现
            matches = np.sum(np.equal(idxpair, nosingleOccurrences[i, :]), axis=1)
            # 如果有元素在其他行出现过，则删除该行
            keepRows[np.where(matches == 1)[0]] = False

        # 根据逻辑索引保留需要的行
        result = idxpair[keepRows, :]
        # 删除idxpair重复的行
        idxpair = np.unique(result, axis=0)
        # 元胞node_distance每一行表示两条线段的一个延长，第一个元素表示第一条线段的起点
        # 第二个元素表示第一条线段的终点；第三个元素表示第二条线段的起点
        # 第四个元素表示第二条线段的终点
        # 延长是线段以起点，朝着终点的方向延长，直至相交
        # 元胞sololocation每一行表示一对孤立点的两个点的位置，1表示点在矩阵edge_line2_A
        # 2表示点在矩阵edge_line2_B
        node_distance = [None] * idxpair.shape[0]
        sololocation = [None] * idxpair.shape[0]
        for i in range(idxpair.shape[0]):
            node_distance[i] = [None] * 4
            sololocation[i] = [None] * 2
            for j in range(edge_line2_A.shape[0]):
                if np.array_equal(edge_line2_A[j, :], solopoints[idxpair[i, 0], :]):
                    sololocation[i][0] = [1, j]
                    node_distance[i][0] = edge_line2_B[j, :]

            for j in range(edge_line2_B.shape[0]):
                if np.array_equal(edge_line2_B[j, :], solopoints[idxpair[i, 0], :]):
                    sololocation[i][0] = [2, j]
                    node_distance[i][0] = edge_line2_A[j, :]

            node_distance[i][1] = solopoints[idxpair[i, 0], :]
            for j in range(edge_line2_A.shape[0]):
                if np.array_equal(edge_line2_A[j, :], solopoints[idxpair[i, 1], :]):
                    sololocation[i][1] = [1, j]
                    node_distance[i][2] = edge_line2_B[j, :]

            for j in range(edge_line2_B.shape[0]):
                if np.array_equal(edge_line2_B[j, :], solopoints[idxpair[i, 1], :]):
                    sololocation[i][1] = [2, j]
                    node_distance[i][2] = edge_line2_A[j, :]

            node_distance[i][3] = solopoints[idxpair[i, 1], :]

        # 找出轮廓线的孤立点的方向
        rays = np.zeros((n, 4))
        for i in range(n):
            psindex = np.all(XYall == solopoints[i, :], axis=1)
            ps = np.where(psindex)[0]
            point1 = [Xall_point[ps, 0], Yall_point[ps, 0]]
            point2 = [Xall_point[ps, 1], Yall_point[ps, 1]]  # 一条线段的两个端点
            point1 = np.array(point1).T
            point2 = np.array(point2).T
            Vector1 = (point2 - point1) / np.linalg.norm(point2 - point1)  # 一条线段的单位方向向量
            point3 = [Xall_point[ps, 2], Yall_point[ps, 2]]
            point4 = [Xall_point[ps, 3], Yall_point[ps, 3]]  # 另一条线段的两个端点
            point3 = np.array(point3).T
            point4 = np.array(point4).T
            Vector2 = (point4 - point3) / np.linalg.norm(point4 - point3)  # 另一条线段的单位方向向量

            point5 = np.full((1, 2), np.nan)
            for ex in range(len(node_distance)):  # 找配对的孤立点
                for ey in range(len(node_distance[ex])):
                    if np.array_equal(node_distance[ex][ey], solopoints[i, :]):
                        point5 = node_distance[ex][ey - 1]
                        break  # 找到后可以跳出循环

                if not np.isnan(np.sum(point5)):
                    break  # 找到目标后退出外部循环

            point6 = solopoints[i, :]  # 已有线段的两个端点
            Vector3 = (point6 - point5) / np.linalg.norm(point6 - point5)  # 已有线段的单位方向向量
            # Vector3与Vector2方向相同或者相反
            if (np.abs(Vector3[0] + Vector2[0, 0]) + np.abs(Vector3[1] + Vector2[0, 1])) < 1e-7 or \
                    (np.abs(Vector3[0] - Vector2[0, 0]) + np.abs(Vector3[1] - Vector2[0, 1])) < 1e-7:
                rays[i, :] = np.hstack((solopoints[i, :].reshape(1, -1), np.unique(Vector1, axis=0)))
            else:
                rays[i, :] = np.hstack((solopoints[i, :].reshape(1, -1),
                                        np.unique(Vector2, axis=0)))  # rays的第1,2列是孤立点的xy坐标，第3,4列是延长的单位方向向量

        # 配对的孤立点的直线延长线是否有交点
        Pcross = np.full((rays.shape[0], 2), np.nan)
        Pe = -1 * np.ones((rays.shape[0], 1))  # 已经配对延长的点在rays的位置
        for ip in range(rays.shape[0]):
            if ip not in Pe:
                Pe = np.append(Pe, ip)
                # 定义第一条直线延长线
                x1, y1, dx1, dy1 = rays[ip, :]
                # 定义第二条直线延长线(配对的孤立点)
                # 寻找配对的孤立点
                rayspyindex = np.all(solopoints == rays[ip, :2], axis=1)
                rayspy = np.where(rayspyindex)[0][0]
                row, col = np.where(idxpair == rayspy)
                if col == 0:
                    ot = idxpair[row, 1]
                else:
                    ot = idxpair[row, 0]

                Pe = np.append(Pe, ot)
                x2, y2, dx2, dy2 = rays[ot[0], :]
                # 求解延长后的新交点-Pcross
                # 解线性方程组，求解 t 和 s
                t = ((x2 - x1) * dy2 - (y2 - y1) * dx2) / (dx1 * dy2 - dx2 * dy1)
                s = ((x2 - x1) * dy1 - (y2 - y1) * dx1) / (dx1 * dy2 - dx2 * dy1)
                if np.isfinite(t) and np.isfinite(s):
                    # 保存交点坐标
                    Pcross[ip, 0] = x1 + t * dx1
                    Pcross[ip, 1] = y1 + t * dy1
                    # 判断延长后的新交点-Pcross是否在四边形内
                    inall = np.full((node_data.shape[0], 1), np.nan)
                    con2 = np.full((node_data.shape[0], 1), np.nan)
                    for i in range(node_data.shape[0]):
                        # 每条线段的多边形的四个顶点的xy坐标
                        edge_node = node_data[i, :][np.newaxis, :, :]
                        # 初始化 x 和 y 列向量
                        x_coords = np.zeros((edge_node.shape[0], edge_node.shape[1]))
                        y_coords = np.zeros((edge_node.shape[0], edge_node.shape[1]))
                        # 循环遍历 edge_node1 中的每个元素，提取 x 坐标和 y 坐标
                        for h in range(edge_node.shape[1]):
                            x_coords[0, h] = edge_node[0, h][0]
                            y_coords[0, h] = edge_node[0, h][1]

                        # 四边形的四个顶点
                        xv = x_coords.reshape(-1, 1)
                        yv = y_coords.reshape(-1, 1)
                        # 当多边形是封闭的，一定首尾相接
                        xv = np.vstack((xv, xv[0]))
                        yv = np.vstack((yv, yv[0]))
                        # 待判断点的横坐标和纵坐标
                        xq = Pcross[ip, 0]
                        yq = Pcross[ip, 1]
                        # 判断点是否在四边形内或线上
                        # in是在四边形内，on是在四边形线上
                        # 添加容错度的方法：将多边形的边界扩展一个小量
                        tolerance = 1e-100  # 设置容错度
                        xv_extended = xv.flatten() + tolerance
                        yv_extended = yv.flatten() + tolerance
                        # 创建 Path 对象
                        polygon_path = Path(np.column_stack((xv_extended, yv_extended)))
                        # 判断点是否在四边形内或线上
                        in_val = polygon_path.contains_points(np.column_stack((xq, yq)))
                        inall[i] = in_val
                        # 考虑到计算机中使用的是有限精度的浮点数表示方法，因此在比较浮点数时会出现舍入误差
                        # 导致出现错误判断两条线段的交点在四边形内
                        # 如果判断点P到多边形线段AB的距离小于或等于1e-7，inpolygon函数将返回1，否则返回0
                        di = np.zeros(((xv.shape[0] - 1), 1))
                        for we in range(xv.shape[0] - 1):
                            xa, ya = xv[we, 0], yv[we, 0]
                            xb, yb = xv[we + 1, 0], yv[we + 1, 0]
                            xp, yp = xq, yq
                            di[we, 0] = np.abs((xb - xa) * (ya - yp) - (xa - xp) * (yb - ya)) / np.sqrt(
                                (xb - xa) ** 2 + (yb - ya) ** 2)

                        if any(di <= 1e-7):
                            con2[i] = 1
                        else:
                            con2[i] = 0

                    # 删除在四边形内的点
                    incon = np.column_stack((inall, con2))
                    if np.any(np.all(incon == np.array([1, 0]), axis=1)):
                        Pcross[ip, :] = np.nan
                        # 如果配对的孤立点的直线延长线交点在四边形包围线内
                        # 分别求孤立点的直线延长线与所有包围线的线段的交点
                        # 已知直线的方向向量和直线上的一点
                        # 定义第一条直线延长线
                        x1, y1, dx1, dy1 = rays[ip, :]
                        direction1 = [dx1, dy1]  # 第一条直线的方向向量
                        P1 = [x1, y1]  # 第一条直线上的一点
                        # 把第一条直线延长为线段
                        tmin, tmax = -shift_vec * 10, shift_vec * 10
                        xmin1, xmax1 = P1[0] + direction1[0] * tmin, P1[0] + direction1[0] * tmax  # 第一条直线的一个端点
                        ymin1, ymax1 = P1[1] + direction1[1] * tmin, P1[1] + direction1[1] * tmax  # 第一条直线的另一个端点
                        # 定义第二条直线延长线(配对的孤立点)
                        x2, y2, dx2, dy2 = rays[ot[0], :]
                        direction2 = [dx2, dy2]  # 另一条直线的方向向量
                        P2 = [x2, y2]  # 另一条直线上的一点
                        # 把另一条直线延长为线段
                        xmin2, xmax2 = P2[0] + direction2[0] * tmin, P2[0] + direction2[0] * tmax  # 第一条直线的一个端点
                        ymin2, ymax2 = P2[1] + direction2[1] * tmin, P2[1] + direction2[1] * tmax  # 第一条直线的另一个端点
                        intersection1 = []
                        fg1 = 0
                        intersection2 = []
                        fg2 = 0
                        # 所有包围线的线段
                        for xc in range(edge_line2.shape[0]):
                            A = edge_line2[xc, :2]  # 线段的一个端点
                            B = edge_line2[xc, 2:]  # 线段的另一个端点
                            # 求与第一条直线的交点的x,y坐标
                            line1 = LineString([(xmin1, ymin1), (xmax1, ymax1)])
                            line2 = LineString([(A[0], A[1]), (B[0], B[1])])
                            intersection = line1.intersection(line2)
                            intx1, inty1 = intersection.xy
                            # 求与另一条直线的交点的x,y坐标
                            line3 = LineString([(xmin2, ymin2), (xmax2, ymax2)])
                            line4 = LineString([(A[0], A[1]), (B[0], B[1])])
                            intersection = line3.intersection(line4)
                            intx2, inty2 = intersection.xy
                            if len(intx1) != 0:
                                fg1 += 1
                                intersection1.append([intx1[0], inty1[0]])

                            if len(intx2) != 0:
                                fg2 += 1
                                intersection2.append([intx2[0], inty2[0]])

                        # 排除孤立点的直线延长线的交点中自身的点-因保留的位数而不完全相等，不能用setdiff
                        intersection1_all = np.array(intersection1)
                        intersection2_all = np.array(intersection2)
                        mat = np.all(np.round(intersection1, 4) == np.round(P1, 4), axis=1)
                        if np.any(mat == 1):
                            xr = np.where(mat == 1)[0]
                            intersection1_all = np.delete(intersection1_all, xr, axis=0)

                        mat2 = np.all(np.round(intersection2, 4) == np.round(P2, 4), axis=1)
                        if np.any(mat2 == 1):
                            xr2 = np.where(mat2 == 1)[0]
                            intersection2_all = np.delete(intersection2_all, xr2, axis=0)

                        # 找最近的交点-intersection1_min-孤立点1
                        distance1 = np.linalg.norm(intersection1_all - np.array(P1), axis=1)  # 计算两个点之间的距离
                        disindex1 = np.argmin(distance1)
                        intersection1_min = intersection1_all[disindex1]
                        # 找最近的交点-intersection2_min-孤立点2
                        distance2 = np.linalg.norm(intersection2_all - np.array(P2), axis=1)
                        disindex2 = np.argmin(distance2)
                        intersection2_min = intersection2_all[disindex2]
                        # 判断孤立点新产生的包围线线段与原始的边是否相交
                        interedge1 = []
                        fg1 = 0
                        interedge2 = []
                        fg2 = 0
                        for ia in range(Edges.shape[0]):
                            pt1 = Coordinates[Edges[ia, 0], :]
                            pt2 = Coordinates[Edges[ia, 1], :]
                            line1 = LineString([(P1[0], P1[1]), (intersection1_min[0], intersection1_min[1])])
                            line2 = LineString([(pt1[0], pt1[1]), (pt2[0], pt2[1])])
                            intersection = line1.intersection(line2)
                            intx1, inty1 = intersection.xy
                            if len(intx1) != 0:
                                fg1 += 1
                                interedge1.append([intx1[0], inty1[0]])

                            line3 = LineString([(P2[0], P2[1]), (intersection2_min[0], intersection2_min[1])])
                            line4 = LineString([(pt1[0], pt1[1]), (pt2[0], pt2[1])])
                            intersection = line3.intersection(line4)
                            intx2, inty2 = intersection.xy
                            if len(intx2) != 0:
                                fg2 += 1
                                interedge2.append([intx2[0], inty2[0]])

                        # 如果孤立点1新产生的包围线线段与原始的所有边都没有相交，优先选择孤立点P1对应的交点作为新的轮廓线的转折点
                        sololocation = np.array(sololocation)
                        interedge1 = np.array(interedge1)
                        if interedge1.size == 0:
                            # 孤立点P1配对的孤立点P2在XY_need的位置换成P1延长后的新交点intersection1_min
                            matchidx = np.all(XY_need == P2, axis=1)
                            xrow = np.where(matchidx)[0]
                            XY_need[xrow, :] = intersection1_min
                            # 孤立点P1配对的孤立点P2在edge_line2的位置sololocation换成P1延长后的新交点intersection1_min
                            matchingRows = np.all(solopoints == P2, axis=1)
                            rowlo = np.where(matchingRows)[0]
                            row, col = np.where(idxpair == rowlo)
                            lo = sololocation[row[0], col[0]]
                            edge_line2[lo[1], lo[0] * 2 - 2:lo[0] * 2] = intersection1_min
                            # 新产生的包围线的线段
                            edge_line2 = np.vstack(
                                [edge_line2, [P1[0], P1[1], intersection1_min[0], intersection1_min[1]]])
                        else:
                            # 孤立点P2配对的孤立点P1在XY_need的位置换成P2延长后的新交点intersection2_min
                            matchidx = np.all(XY_need == P1, axis=1)
                            xrow = np.where(matchidx)[0]
                            XY_need[xrow, :] = intersection2_min
                            # 孤立点P2配对的孤立点P1在edge_line2的位置sololocation换成P2延长后的新交点intersection2_min
                            matchingRows = np.all(solopoints == P1, axis=1)
                            rowlo = np.where(matchingRows)[0]
                            row, col = np.where(idxpair == rowlo)
                            lo = sololocation[row[0], col[0]]
                            edge_line2[lo[1], lo[0] * 2 - 2:lo[0] * 2] = intersection2_min
                            # 新产生的包围线的线段
                            edge_line2 = np.vstack(
                                [edge_line2, [P2[0], P2[1], intersection2_min[0], intersection2_min[1]]])

                    else:
                        XY_need = np.vstack([XY_need, [Pcross[ip, 0], Pcross[ip, 1]]])
                        edge_line2 = np.vstack([edge_line2, [rays[ip, 0], rays[ip, 1], Pcross[ip, 0], Pcross[ip, 1]]])
                        edge_line2 = np.vstack(
                            [edge_line2, [rays[ot[0], 0], rays[ot[0], 1], Pcross[ip, 0], Pcross[ip, 1]]])

                    # 配对的孤立点在同一直线上时，直接连接这两个孤立点
                else:
                    edge_line2 = np.vstack([edge_line2, [rays[ip, 0], rays[ip, 1], rays[ot, 0], rays[ot, 1]]])

        # 删除重复的点和线段(同样的延长线产生的交点会因保留的位数而不完全相等，统一保留至小数点后4位)
        XY_need = np.unique(np.round(XY_need, 4), axis=0)
        edge_line2 = np.unique(np.round(edge_line2, 4), axis=0)
        ## 排列转折点和轮廓线
        # 顺时针顺序的转折点是XY_need2
        # 找到起点（x最小中y最大的点)-nodefirst
        XY_need2 = np.zeros_like(XY_need)
        minx = np.min(XY_need[:, 0])
        rowmin = np.where(XY_need[:, 0] == minx)[0]
        maxy = np.max(XY_need[rowmin, 1])
        nodefirst = np.array([minx, maxy]).reshape((1, 2))

        nodet = 1
        while nodet <= XY_need.shape[0]:
            # 轮廓线以起点作为第一个点，依次连接
            linesqe = []
            for i in range(edge_line2.shape[0]):
                if np.array_equal(edge_line2[i, :2].reshape((1, 2)), nodefirst):
                    linesqe.append(i)

            # 如果起点都在edge_line2的第一，二列
            if len(linesqe) == 2:
                nodepair = edge_line2[linesqe, 2:4]
                # 起点的一个位置是在edge_line2的第一，二列，另一个位置是在edge_line2的第三，四列
            elif len(linesqe) == 1:
                for i in range(edge_line2.shape[0]):
                    if np.array_equal(edge_line2[i, 2:4].reshape((1, 2)), nodefirst):
                        linesqe.append(i)

                nodepair = np.vstack((edge_line2[linesqe[0], 2:4], edge_line2[linesqe[1], 0:2]))
                # 起点的两个位置都是在edge_line2的第三，四列
            else:
                for i in range(edge_line2.shape[0]):
                    if np.array_equal(edge_line2[i, 2:4].reshape((1, 2)), nodefirst):
                        linesqe.append(i)

                nodepair = edge_line2[linesqe, 0:2]

            # 顺时针连接点
            # 当是第一个点的时候，此时连接点是x最大的点
            if nodet == 1:
                idxcon = np.argmax(nodepair[:, 0])
                nodeend = nodepair[idxcon, :].reshape((1, 2))
                # 当是最后一个点，此时连接点是第一个点
            elif nodet == XY_need.shape[0]:
                nodeend = nodefirst
                # 当是其它点，此时连接点是另一个没有连接的点
            else:
                # nodeend=setdiff(nodepair,XY_need2(nodet-1,:),'rows');
                np2 = XY_need2[nodet - 2, :]
                nodepair_str = np.array([' '.join(map(str, row)) for row in nodepair], dtype='str')
                np2_str = np.array([' '.join(map(str, row)) for row in [np2]], dtype='str')
                nodeend_str = np.setdiff1d(nodepair_str, np2_str, assume_unique=True)
                nodeend = np.array([list(map(float, row.split())) for row in nodeend_str])

            XY_need2[nodet - 1, :] = nodefirst
            nodet += 1
            nodefirst = nodeend

        # 检测转折点的连接顺序是否正确-XY_need3
        # 删除XY_need2重复的行但不能改变其他行的位置，因为位置代表连接顺序（unique会按照第一列的递增顺序排列）
        _, idx = np.unique(XY_need2, axis=0, return_index=True)
        XY_need2 = XY_need2[np.sort(idx), :]
        XY_need3 = np.vstack((XY_need2, [minx, maxy]))
        plt.figure()
        plt.plot(XY_need3[:, 0], XY_need3[:, 1], 'k-', linewidth=2)
        plt.plot(XY_need3[:, 0], XY_need3[:, 1], 'r*')
        plt.axis('equal')
        # 把包围线和原始图放在一起，形成更鲜明的对比
        # 绘制宽度=10的多边形
        for i in range(Edges.shape[0]):
            pt1 = Coordinates[Edges[i, 0], :]
            pt2 = Coordinates[Edges[i, 1], :]
            plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'g-', linewidth=2)

        plt.axis('equal')
        plt.title('Edge Polygon and Original Polygon')
        plt.show()

        # 计算包围线内的面积
        totalArea = 0  # 初始化总面积
        # 计算每个三角形的面积并相加
        for i in range(XY_need3.shape[0] - 1):
            x1, y1 = XY_need3[i, 0], XY_need3[i, 1]
            x2, y2 = XY_need3[i + 1, 0], XY_need3[i + 1, 1]
            # 计算三角形的面积并累加
            triangleArea = 0.5 * (x1 * y2 - x2 * y1)
            totalArea += triangleArea

        # 面积的绝对值即为多边形的面积
        polygonArea = np.abs(totalArea)

        if userInput1 == 1:
            # 将double 数组转换为table
            df = pd.DataFrame(XY_need3, columns=['X_coordinate', 'Y_coordinate'])
            # 将文件夹名添加到文件路径
            outputFile = os.path.join(foldname, 'LGT Zone Boundary.xlsx')
            # 检查文件是否存在并删除
            try:
                os.remove(outputFile)  # 如果文件存在，删除它
            except FileNotFoundError:
                pass

            # 将 DataFrame 保存为 Excel 文件
            df.to_excel(outputFile, index=False)

        # 所有统计参数写成struct结构，用一个变量传递数据
        resultedge = {'XY': XY_need3, 'area': polygonArea}

        return resultedge, XY_need3

    def Lighting_Parameters_Distribution(self, MC_lgtn, DSave, Line, resultedge, foldname):
        # 绘图
        plt.close('all')

        # struct获取每个变量名
        mode = MC_lgtn['mode']
        muN = MC_lgtn['muN']
        sigmaN = MC_lgtn['sigmaN']
        rhoi = MC_lgtn['rhoi']
        fixn = MC_lgtn['fixn']
        muI = MC_lgtn['muI']
        mu = MC_lgtn['mu']
        sigma1st = MC_lgtn['sigma1st']
        sigma = MC_lgtn['sigma']
        userInput2 = DSave['userInput2']
        userInput0 = DSave['userInput0']
        Coordinates = Line['Node']
        Edges = Line['Edge'].astype(int)
        XY_need3 = resultedge['XY']

        # python特定排序从0开始
        Edges -= 1

        # fixed total number of flashes
        if mode == 1:
            flash = fixn
            sN_all = np.zeros((1, flash), dtype=int)
            for e in range(flash):
                ZkN = np.random.randn(1)  # ZkN is a standard normal variates.(随机生成的标准正态变量)
                sN = np.round(np.exp(muN + sigmaN * ZkN))  # 四舍五入
                # number of stroke(stroke的范围是[1,20])
                if sN > 20:
                    sN = np.array([20.])
                elif sN < 1:
                    sN = np.array([1.])

                sN_all[0, e] = sN[0]

        if mode == 2:
            # fixed total number of stroks(设置超过fixn次停止)
            flash_init = fixn
            sN_all_init = np.zeros((1, flash_init), dtype=int)
            for e in range(flash_init):
                ZkN = np.random.randn(1)  # ZkN is a standard normal variates.(随机生成的标准正态变量)
                sN_init = np.round(np.exp(muN + sigmaN * ZkN))  # 四舍五入
                # number of stroke(stroke的范围是[1,20])
                if sN_init > 20:
                    sN_init = np.array([20.])
                elif sN_init < 1:
                    sN_init = np.array([1.])

                sN_all_init[0, e] = sN_init[0]

            for e in range(flash_init):
                stroke_sum = np.sum(sN_all_init[0, 0:e + 1])
                if stroke_sum > fixn:
                    break

            flash = e + 1
            sN_all = sN_all_init[0, 0:e + 1]

        flash_number = np.repeat(np.arange(1, sN_all.size + 1), sN_all)  # flash_number = flash_number.reshape(-1,1).T
        stroke_number = np.concatenate([np.arange(1, s + 1) for s in sN_all])

        # 画第一张图 - 杆塔图
        # 绘制宽度=10的多边形
        for t in range(4):
            plt.figure()
            for i in range(len(Edges)):
                pt1 = Coordinates[Edges[i, 0], :]
                pt2 = Coordinates[Edges[i, 1], :]
                plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'k-', linewidth=2)
            plt.scatter(Coordinates[:, 0], Coordinates[:, 1], s=50, color='k', marker='o')  # 画杆塔代表的点

        # plt.show()
        # 为第二步画最小的长方形做准备-随机形成uniform分布的点
        # 最小的x坐标到最大的x坐标
        xmin = np.min(XY_need3[:, 0])  # 左下角x坐标
        xmax = np.max(XY_need3[:, 0])  # 右上角x坐标
        ymin = np.min(XY_need3[:, 1])  # 左下角y坐标
        ymax = np.max(XY_need3[:, 1])  # 右上角y坐标
        # 画第2张图-杆塔图+包围线
        plt.figure(2)
        plt.plot(XY_need3[:, 0], XY_need3[:, 1], 'k-', linewidth=2)
        plt.plot(XY_need3[:, 0], XY_need3[:, 1], 'r*')
        plt.axis('equal')
        pn = fixn * 100  # 点的数量

        xp = xmin + (xmax - xmin) * np.random.rand(pn, 1)  # 生成x坐标
        yp = ymin + (ymax - ymin) * np.random.rand(pn, 1)  # 生成y坐标

        # 判断随机点是否在包围线内
        # 包围线的顶点(当多边形是封闭的，一定首尾相接)-XY_need3
        xv = XY_need3[:, 0]
        yv = XY_need3[:, 1]

        # 如果同一个flash的第1个stroke在包围线外，去掉整个flash原本的点. 但维持flash or stroke总数不变
        # 如果同一个flash的第1个stroke在包围线内，同一个flash的第1个stroke的点等于同一个flash所有stroke的点
        points_need = []  # 在包围线内的点
        firstroke = np.where(stroke_number == 1)[0]
        ei = 0
        while ei < len(firstroke):
            if ei != len(firstroke) - 1:
                fei = firstroke[ei]
                feinext = firstroke[ei + 1]
                # 待判断点的横坐标-xp(fei)和纵坐标-yp(fei)
                # 判断点是否在四边形内或线上
                # in是在四边形内，on是在四边形线上
                in_poly = Path(list(zip(xv, yv))).contains_point((xp[ei], yp[ei]))
                if in_poly:
                    points_need.extend([xp[ei], yp[ei]] * (feinext - fei))
                    ei += 1
                else:
                    xp = np.delete(xp, np.arange(ei, (feinext - fei + ei))).reshape(-1, 1)
                    yp = np.delete(yp, np.arange(ei, (feinext - 1 - fei + ei))).reshape(-1, 1)
                    ei = ei

            else:
                fei = firstroke[ei]
                feinext = len(stroke_number) - 1
                # 待判断点的横坐标-xp(fei)和纵坐标-yp(fei)
                # 判断点是否在四边形内或线上
                # in是在四边形内，on是在四边形线上
                in_poly = Path(list(zip(xv, yv))).contains_point((xp[ei], yp[ei]))
                if in_poly:
                    points_need.extend([xp[ei], yp[ei]] * (feinext - fei + 1))
                    ei += 1
                else:
                    xp = np.delete(xp, np.arange(ei, (feinext - fei + ei + 1))).reshape(-1, 1)
                    yp = np.delete(yp, np.arange(ei, (feinext - fei + ei + 1))).reshape(-1, 1)
                    ei = ei

        # 每个flash总共有多少次stroke
        flash_counts = Counter(flash_number)
        unique_flash = np.array(list(flash_counts.keys()))  # 注意是一维
        stroke_counts = np.array(list(flash_counts.values()))

        # 第3张图-杆塔图+包围线+在包围线内的点-points_need
        points_need = np.array(points_need).reshape(-1, 2)
        plt.figure(3)
        plt.scatter(points_need[:, 0], points_need[:, 1], s=1, color='r', marker='o')
        # plt.show()

        # the Monte Carlo
        # Number of samples
        n_samples = len(stroke_number)
        # Initialize variables:
        Ip = np.zeros(n_samples)
        tf = np.zeros(n_samples)
        Sm = np.zeros(n_samples)
        th = np.zeros(n_samples)
        lightp = np.zeros((n_samples, 4))
        # Lighting parameters: Ip tf Sm th
        # 1st stroke
        i = 0
        while i < n_samples:
            stroke_present = stroke_number[i]
            if stroke_present == 1:
                rho = rhoi
                # Generate the symmetric part
                rho = rho + np.triu(rho, 1).T
                np.fill_diagonal(rho, 1)
                # the symmetric covariance matrix K
                # The ij-th off-diagonal element of K is given by correlation coefficient ρij between xi and xj
                # multiplied by the product of their two corresponding standard deviations (i.e., σxi and σxj)
                # whilst the ii-th diagonal element is equal to variance σxi^2 of random variable xi
                K = np.zeros((4, 4))
                for n in range(4):
                    for j in range(4):
                        K[n, j] = rho[n, j] * sigma1st[n] * sigma1st[j]

                # Let be Q = K^(-1)
                Q = np.linalg.inv(K)
                # the conditional variance of xn
                sigmaco = np.zeros(4)
                for b in range(1, 4):
                    sigmaco[b] = np.sqrt(1 / Q[b, b])

                # the conditional mean of xn
                muco = np.zeros(4)
                # Zk , Zk+1, Zk+2, Zk+3 are four standard normal variates.(随机生成的标准正态变量)
                Zk = np.random.randn(4)
                # Step 1) for the calculation of an Ip value：
                Ip[i] = np.exp(muI[0] + sigma1st[0] * Zk[0])
                # Step 2) for the calculation of a tf value
                # the conditional mean of x2
                muco[1] = muI[1] - (1 / Q[1, 1]) * (Q[1, 0] * (np.log(Ip[i]) - muI[0]))
                tf[i] = np.exp(muco[1] + sigmaco[1] * Zk[1])
                # Step 3) for the calculation of a Sm value：
                muco[2] = muI[2] - (1 / Q[2, 2]) * (
                            Q[2, 0] * (np.log(Ip[i]) - muI[0]) + Q[2, 1] * (np.log(tf[i]) - muI[1]))
                Sm[i] = np.exp(muco[2] + sigmaco[2] * Zk[2])
                # Step 4) for the calculation of a th value：
                muco[3] = muI[3] - (1 / Q[3, 3]) * (
                        Q[3, 0] * (np.log(Ip[i]) - muI[0]) + Q[3, 1] * (np.log(tf[i]) - muI[1]) + Q[3, 2] * (
                        np.log(Sm[i]) - muI[2]))
                th[i] = np.exp(muco[3] + sigmaco[3] * Zk[3])
                # Ip的范围是[3,200] , tf的范围是[0,30] , Sm的范围是[0,200] and th的范围是[0,500]???????????????
                valid_Ip = (3 <= Ip[i] <= 200)  # 判断Ip范围
                valid_tf = (1e-3 < tf[i] <= 30)  # 判断tf范围
                valid_Sm = (1e-3 < Sm[i] <= 200)  # 判断Sm范围
                valid_th = (1e-3 < th[i] <= 500)  # 判断th范围
                i += valid_Ip * valid_tf * valid_Sm * valid_th

                # a given quadruple of values for Ip tf Sm th
                lightp = np.vstack((Ip, tf, Sm, th)).T
            else:
                rho = rhoi
                # Generate the symmetric part
                rho = rho + np.triu(rho, 1).T
                np.fill_diagonal(rho, 1)
                # the symmetric covariance matrix K
                # The ij-th off-diagonal element of K is given by correlation coefficient ρij between xi and xj
                # multiplied by the product of their two corresponding standard deviations (i.e., σxi and σxj)
                # whilst the ii-th diagonal element is equal to variance σxi^2 of random variable xi
                K = np.zeros((4, 4))
                for n in range(4):
                    for j in range(4):
                        K[n, j] = rho[n, j] * sigma[n] * sigma[j]

                # Let be Q = K^(-1)
                Q = np.linalg.inv(K)
                # the conditional variance of xn
                sigmaco = np.zeros(4)
                for b in range(1, 4):
                    sigmaco[b] = np.sqrt(1 / Q[b, b])

                # the conditional mean of xn
                muco = np.zeros(4)
                # Zk , Zk+1, Zk+2, Zk+3 are four standard normal variates.(随机生成的标准正态变量)
                Zk = np.random.randn(4)
                # Step 1) for the calculation of an Ip value：
                Ip[i] = np.exp(mu[0] + sigma[0] * Zk[0])
                # Step 2) for the calculation of a tf value
                # the conditional mean of x2
                muco[1] = mu[1] - (1 / Q[1, 1]) * (Q[1, 0] * (np.log(Ip[i]) - mu[0]))
                tf[i] = np.exp(muco[1] + sigmaco[1] * Zk[1])
                # Step 3) for the calculation of a Sm value：
                muco[2] = mu[2] - (1 / Q[2, 2]) * (
                            Q[2, 0] * (np.log(Ip[i]) - mu[0]) + Q[2, 1] * (np.log(tf[i]) - mu[1]))
                Sm[i] = np.exp(muco[2] + sigmaco[2] * Zk[2])
                # Step 4) for the calculation of a th value：
                muco[3] = mu[3] - (1 / Q[3, 3]) * (
                        Q[3, 0] * (np.log(Ip[i]) - mu[0]) + Q[3, 1] * (np.log(tf[i]) - mu[1]) + Q[3, 2] * (
                        np.log(Sm[i]) - mu[2]))
                th[i] = np.exp(muco[3] + sigmaco[3] * Zk[3])
                # Ip的范围是[3,200] , tf的范围是[0,30] , Sm的范围是[0,200] and th的范围是[0,500]???????????????
                valid_Ip = (3 <= Ip[i] <= 200)  # 判断Ip范围
                valid_tf = (1e-3 < tf[i] <= 30)  # 判断tf范围
                valid_Sm = (1e-3 < Sm[i] <= 200)  # 判断Sm范围
                valid_th = (1e-3 < th[i] <= 500)  # 判断th范围
                i += valid_Ip * valid_tf * valid_Sm * valid_th

                # a given quadruple of values for Ip tf Sm th
                lightp = np.vstack((Ip, tf, Sm, th)).T

        parameterst = np.hstack((flash_number.reshape(-1, 1), stroke_number.reshape(-1, 1), lightp))

        # 每一个flash的第一次stroke的位置
        siteone = np.where(parameterst[:, 1] == 1)[0]

        if userInput2 == 1:
            # 将double 数组转换为table
            df2 = pd.DataFrame(parameterst, columns=['flash', 'stroke', 'Ip', 'tf', 'Sm', 'th'])
            # 将文件夹名添加到文件路径
            outputFile = os.path.join(foldname, 'Current Parameter.xlsx')
            # 检查文件是否存在并删除
            try:
                os.remove(outputFile)  # 如果文件存在，删除它
            except FileNotFoundError:
                pass

            # 将 DataFrame 保存为 Excel 文件
            df2.to_excel(outputFile, index=False)
            # save 有哪些次flash and 每次flash有多少个stroke
            flash_stroke = np.column_stack((unique_flash, stroke_counts))
            # # 将double 数组转换为table
            # df27 = pd.DataFrame(flash_stroke, columns=['flash', 'stroke'])
            # # 将文件夹名添加到文件路径
            # outputFile = os.path.join(foldname, 'Flash_Stroke Dist.xlsx')
            # # 检查文件是否存在并删除
            # try:
            #     os.remove(outputFile)  # 如果文件存在，删除它
            # except FileNotFoundError:
            #     pass
            #
            # # 将 DataFrame 保存为 Excel 文件
            # df27.to_excel(outputFile, index=False)

        if userInput0 == 1:
            # 将double 数组转换为table
            df0 = pd.DataFrame(points_need, columns=['X_coordinate', 'Y_coordinate'])
            # 将文件夹名添加到文件路径
            outputFile = os.path.join(foldname, 'Stroke Location.xlsx')
            # 检查文件是否存在并删除
            try:
                os.remove(outputFile)  # 如果文件存在，删除它
            except FileNotFoundError:
                pass

            # 将 DataFrame 保存为 Excel 文件
            df0.to_excel(outputFile, index=False)

        # 所有统计参数写成struct结构，用一个变量传递数据
        resultcur = {'parameterst': parameterst,
                     'points_need': points_need,
                     'siteone': siteone,
                     'unique_flash': unique_flash,  # 有哪些次flash
                     'stroke_counts': stroke_counts}  # 每次flash有多少个stroke

        return resultcur,parameterst,flash_stroke,points_need

    def Lightning_Stroke_Location(self, Line, resultcur, DSave, foldname, resultedge):
        # struct获取每个变量名
        Coordinates = Line['Node']
        Edges = Line['Edge'].astype(int)
        points_need = resultcur['points_need']
        parameterst = resultcur['parameterst']
        siteone = resultcur['siteone']
        unique_flash = resultcur['unique_flash']
        stroke_counts = resultcur['stroke_counts']
        polygonArea = resultedge['area']
        OHLPf = Line['OHLPf']
        segments = Line['segments']
        SWnumber = Line['SWnumber']
        userInput3 = DSave['userInput3']
        userInput3d = DSave['userInput3d']
        userInput3ind = DSave['userInput3ind']
     
        # python特定排序从0开始
        Edges -= 1

        # 判断每个点属于哪条线（哪个span）-span_min
        # 记录原始的线段-origin_line
        origin_line = []
        for i in range(Edges.shape[0]):
            origin_line.append([Coordinates[Edges[i, 0], :], Coordinates[Edges[i, 1], :]])

        origin_line = np.array(origin_line)
        # 每一个落雷点
        span_min = []  # 第1列是shield wire的端点；第2列是shield wire的端点；第3列是shield wire的高度
        span_pc = []  # 第1(4)列是phase conductor的端点；第2(5)列是phase conductor的端点；第3(6)列是phase conductor的高度
        span_np = []  # pc对应#相
        circle = []
        conductor = []
        esegment = np.zeros(points_need.shape[0]).astype(int)
        dmin_all = np.zeros(points_need.shape[0])  # 每一个落雷点最近的边的编号
        for tp in range(points_need.shape[0]):
            P = points_need[tp, :]  # 待判断的落雷点
            d = np.zeros(len(origin_line))
            # 每一条线段
            for to in range(len(origin_line)):
                A, B = origin_line[to][0], origin_line[to][1]  # 线段的端点
                d[to] = self.Point_To_Line_Distance(A, B, P)

            dmin_index = np.argmin(d)
            dmin_all[tp] = dmin_index
            # segments = np.array(segments[0, dmin_index])
            esegment[tp] = segments[0, dmin_index]
            span0 = OHLPf[dmin_index][0]
            span_np.append(OHLPf[dmin_index][1])
            circle.append(OHLPf[dmin_index][2])
            conductor.append(OHLPf[dmin_index][3])

            # sw有2根
            if SWnumber[0, dmin_index] == 2:
                # 第1,2列是sw的第1根的端点,第3列是sw的第1根的高度                  第1,2列是sw的第2根的端点,第3列是sw的第2根的高度
                span_min.append([span0[1, 0:2], span0[1, 3:5], span0[1, 2], span0[2, 0:2], span0[2, 3:5], span0[2, 2]])
            # sw有1根
            elif SWnumber[0, dmin_index] == 1:
                # 第1,2列是sw的端点,第3列是sw的高度
                span_min.append([span0[1, 0:2], span0[1, 3:5], span0[1, 2]])
            # sw有0根
            else:
                span_min.append([])  # 不存在sw

            # pc有2相
            if len(span_np[tp]) == 2:
                # 第1,2列是pc的第1相的端点,第3列是pc的第1相的高度       第4.5列是pc的第2相的端点,第6列是pc的第2相的高度
                span_pc.append([span0[SWnumber[0, dmin_index] + 1, 0:2], span0[SWnumber[0, dmin_index] + 1, 3:5],
                                span0[SWnumber[0, dmin_index] + 1, 2], span0[SWnumber[0, dmin_index] + 2, 0:2],
                                span0[SWnumber[0, dmin_index] + 2, 3:5], span0[SWnumber[0, dmin_index] + 2, 2]])
            # pc有1相
            else:
                # 第1,2列是pc唯一1相的端点,第3列是pc唯一1相的高度
                span_pc.append([span0[SWnumber[0, dmin_index] + 1, 0:2], span0[SWnumber[0, dmin_index] + 1, 3:5],
                                span0[SWnumber[0, dmin_index] + 1, 2]])

        # 每段span的pc对应#相从列的编号换成相的circleID-circlep,相的数字-span_npa
        # 每段span的sw的circleID-circles
        dmin_all = [int(x) for x in dmin_all]
        span_npa = []
        circlep = []
        circles = []
        sw_npa = []
        for fh in range(len(span_np)):
            ev_span_np = span_np[fh]
            ev_span_np = [x - 1 for x in ev_span_np]
            e_circle = circle[fh]
            ev_span_npa = []
            ev_sw_npa = []
            ecirclep = []
            ecircles = []
            a = e_circle[ev_span_np, 1]
            ev_span_npa = e_circle[ev_span_np, 1].T
            ev_sw_npa = e_circle[1:(SWnumber[0, dmin_all[fh]] + 1), 1].T
            ecirclep = e_circle[ev_span_np, 0].T
            ecircles = e_circle[1:(SWnumber[0, dmin_all[fh]] + 1), 0].T

            span_npa.append(ev_span_npa)
            sw_npa.append(ev_sw_npa)
            circlep.append(ecirclep)
            circles.append(ecircles)

        conductor=np.array(conductor[0])
        # EGM model
        # 判断每个点points_need雷击点的位置-stroke_result
        stroke_position = []
        stroke_point = []
        for i in range(points_need.shape[0]):
            P = points_need[i, :]  # 待判断的落雷点
            # lightning current,单位是KA
            I = parameterst[i, 2]  # I=待判断的落雷点的电路参数的Ip
            # sw有0根,pc的吸引半径是rc
            if sum(x is not None for x in span_min[i]) == 0:
                # pc有2相
                if span_npa[i].size == 2:
                    C, D = span_pc[i][:2]  # pc的第1相的起点终点
                    E, F = span_pc[i][3:5]  # pc的第2相的起点终点
                    # 用公式计算吸引半径(sw与不同的pc比较会有不同的吸引半径)
                    yc = np.zeros((1, 2))  # pc的第i相的高度
                    yc[0, 0] = span_pc[i][2]
                    yc[0, 1] = span_pc[i][5]
                    # 用公式计算吸引半径
                    rc = 10 * I ** 0.65
                    dc = [rc, rc]
                    # 矩阵的元素是同一数据类型，stroke_1第一列是高度，第二列是吸引半径;stroke_2是雷击点的位置名称-span_npa
                    stroke_1 = np.array([[yc[0, 0], dc[0], np.nan], [yc[0, 1], dc[1], np.nan]])  # pc的第1相;pc的第2相
                    sc = self.Point_To_Line_Distance(C, D, P)  # 判断点到pc的第1相的距离
                    sb = self.Point_To_Line_Distance(E, F, P)  # 判断点到pc的第2相的距离
                    stroke_2 = [["Direct", "phase conductor", circlep[i][0, 0], span_npa[i][0, 0],conductor[0, 0]],
                                ["Direct", "phase conductor", circlep[i][0, 1], span_npa[i][0, 1],conductor[1, 0]]]
                    # 谁高谁先判断-stroke_1 stroke_3 sall_points
                    indices = np.argsort(-stroke_1[:, 0])
                    stroke_1 = stroke_1[indices, :]
                    stroke_3 = np.array(stroke_2)[indices]
                    sall = np.array([sc, sb])  # 点到pc的第1相,pc的第2相的距离
                    sall = sall[indices]
                    sall_points = np.array([[C, D], [E, F]])  # 跟哪条线段的pc有关系
                    sall_points = sall_points[indices]
                    if sall[0] <= stroke_1[0, 1]:
                        stroke_position.append(stroke_3[0, :])
                        stroke_point.append(sall_points[0, :])
                    elif sall[1] <= stroke_1[1, 1]:
                        stroke_position.append(stroke_3[1, :])
                        stroke_point.append(sall_points[1, :])
                    else:
                        stroke_position.append(np.array(["Indirect", "ground", np.nan, np.nan, np.nan]))
                        stroke_point.append(np.nan)

                    # 当pc的2个相高度一样时，判断两条pc的重叠区，离哪个pc近落在哪个pc
                    if stroke_position[-1][1] == "phase conductor" and yc[0, 0] == yc[0, 1]:
                        # 点到pc的第1相的距离大于点到pc的第2相的距离
                        if sc > sb:
                            stroke_position[-1][2] = circlep[i][0, 1]
                            stroke_position[-1][3] = span_npa[i][0, 1]
                            stroke_position[-1][4] = conductor[1, 0]
                            stroke_point[-1] = np.array([E, F])
                        else:
                            stroke_position[-1][2] = circlep[i][0, 0]
                            stroke_position[-1][3] = span_npa[i][0, 0]
                            stroke_position[-1][4] = conductor[0, 0]
                            stroke_point[-1] = np.array([C, D])

                # pc有1相
                else:
                    C, D = span_pc[i][:2]  # pc唯一1相的起点终点
                    # 用公式计算吸引半径
                    rc = 10 * I ** 0.65
                    dc = rc  # pc的吸引半径
                    sc = self.Point_To_Line_Distance(C, D, P)  # 判断点到pc唯一1相的距离
                    if sc <= dc:
                        stroke_position.append(["Direct", "phase conductor", circlep[i][0, 0], span_npa[i][0, 0]], conductor[0, 0])
                        stroke_point.append(np.array([C, D]))
                    else:
                        stroke_position.append(np.array(["Indirect", "ground", np.nan, np.nan, np.nan]))
                        stroke_point.append(np.nan)

            # sw有1根
            elif sum(x is not None for x in span_min[i]) == 3:
                A, B = span_min[i][:2]  # shield wire的起点终点
                ss = self.Point_To_Line_Distance(A, B, P)  # 判断点到shield wire的距离
                # pc有2相
                if span_npa[i].size == 2:
                    C, D = span_pc[i][:2]  # pc的第1相的起点终点
                    E, F = span_pc[i][3:5]  # pc的第2相的起点终点
                    # 用公式计算吸引半径(sw与不同的pc比较会有不同的吸引半径)
                    da = np.zeros(2)  # pc的第i相与shield wire的距离差
                    da[0] = np.sqrt((C[0] - A[0]) ** 2 + (C[1] - A[1]) ** 2)
                    da[1] = np.sqrt((E[0] - A[0]) ** 2 + (E[1] - A[1]) ** 2)
                    yg = span_min[i][2]  # shield wire的高度
                    yc = np.zeros((1, 2))  # pc的第i相的高度
                    yc[0, 0] = span_pc[i][2]
                    yc[0, 1] = span_pc[i][5]
                    # 用公式计算吸引半径
                    rc = 10 * I ** 0.65
                    rg = 5.5 * I ** 0.65
                    dc = np.zeros((1, 2))  # pc的吸引半径(第1相, 第2相)
                    dg = np.zeros((1, 2))  # sw的吸引半径(对于第1相, 对于第2相)
                    alpha = []
                    theta = []
                    beta = []
                    for xks in range(2):
                        alpha = np.arctan(da[xks] / (yg - yc[0, xks]))  # 单位是弧度,因为三角函数的输入参数默认为弧度
                        theta = np.arcsin((rg - yc[0, xks]) / rc)  # 单位是弧度
                        beta = np.arcsin(((yg - yc[0, xks]) * np.sqrt(1 + np.tan(alpha) ** 2)) / (2 * rc))  # 单位是弧度
                        dc[0, xks] = rc * (np.cos(theta) - np.cos(alpha + beta))
                        dg[0, xks] = rc * np.cos(alpha - beta)  # shield wire的吸引半径

                    # 矩阵的元素是同一数据类型，stroke_1第一列是高度，第二列是吸引半径;stroke_2是雷击点的位置名称-span_npa
                    stroke_1 = np.array([[yg, dg[0, 0], dg[0, 1]], [yc[0, 0], dc[0, 0], np.nan],
                                         [yc[0, 1], dc[0, 1], np.nan]])  # sw;pc的第1相;pc的第2相
                    sc = self.Point_To_Line_Distance(C, D, P)  # 判断点到pc的第1相的距离
                    sb = self.Point_To_Line_Distance(E, F, P)  # 判断点到pc的第1相的距离
                    # 当点离pc的第i相更近，判断sw的吸引半径，先用sw对于第i相的吸引半径
                    if sc > sb:
                        stroke_1[0, :] = np.array([yg, dg[0, 1], np.nan])
                    else:
                        stroke_1[0, :] = np.array([yg, dg[0, 0], np.nan])

                    stroke_2 = [["Direct", "shield wire", circles[i][0], sw_npa[i][0], 1] ,
                                ["Direct", "phase conductor", circlep[i][0, 0], span_npa[i][0, 0], conductor[0, 0]],
                                ["Direct", "phase conductor", circlep[i][0, 1], span_npa[i][0, 1], conductor[1, 0]]]
                    # 谁高谁先判断-stroke_1 stroke_3 sall_points
                    indices = np.argsort(-stroke_1[:, 0])
                    stroke_1 = stroke_1[indices, :]
                    stroke_3 = np.array(stroke_2)[indices]
                    sall = np.array([ss, sc, sb])  # 点到sw,pc的第1相,pc的第2相的距离
                    sall = sall[indices]
                    sall_points = np.array([[A, B], [C, D], [E, F]])  # 跟哪条线段的sw或者pc有关系
                    sall_points = sall_points[indices]
                    if sall[0] <= stroke_1[0, 1]:
                        stroke_position.append(stroke_3[0, :])
                        stroke_point.append(sall_points[0, :])
                    elif sall[1] <= stroke_1[1, 1]:
                        stroke_position.append(stroke_3[1, :])
                        stroke_point.append(sall_points[1, :])
                    elif sall[2] <= stroke_1[2, 1]:
                        stroke_position.append(stroke_3[2, :])
                        stroke_point.append(sall_points[2, :])
                    else:
                        stroke_position.append(np.array(["Indirect", "ground", np.nan, np.nan, np.nan]))
                        stroke_point.append(np.nan)

                    # 当pc的2个相高度一样时，判断两条pc的重叠区，离哪个pc近落在哪个pc
                    if stroke_position[-1][1] == "phase conductor" and yc[0, 0] == yc[0, 1]:
                        # 点到pc的第1相的距离大于点到pc的第2相的距离
                        if sc > sb:
                            stroke_position[-1][2] = circlep[i][0, 1]
                            stroke_position[-1][3] = span_npa[i][0, 1]
                            stroke_position[-1][4] = conductor[1, 0]
                            stroke_point[-1] = np.array([E, F])
                        else:
                            stroke_position[-1][2] = circlep[i][0, 0]
                            stroke_position[-1][3] = span_npa[i][0, 0]
                            stroke_position[-1][4] = conductor[0, 0]
                            stroke_point[-1] = np.array([C, D])

                # pc有1相
                else:
                    C, D = span_pc[i][:2]  # pc唯一1相的起点终点
                    # 用公式计算吸引半径
                    da = np.sqrt((C[0] - A[0]) ** 2 + (C[1] - A[1]) ** 2)  # pc的唯一1相与shield wire的距离差
                    yg = span_min[i][2]  # shield wire的高度
                    yc = span_pc[i][2]  # pc的唯一1相的高度
                    # 用公式计算吸引半径
                    rc = 10 * I ** 0.65
                    rg = 5.5 * I ** 0.65
                    alpha = []
                    theta = []
                    beta = []
                    alpha = np.arctan(da / (yg - yc))  # 单位是弧度,因为三角函数的输入参数默认为弧度
                    theta = np.arcsin((rg - yc) / rc)  # 单位是弧度
                    beta = np.arcsin(((yg - yc) * np.sqrt(1 + np.tan(alpha) ** 2)) / (2 * rc))  # 单位是弧度
                    dc = rc * (np.cos(theta) - np.cos(alpha + beta))  # pc的吸引半径
                    dg = rc * np.cos(alpha - beta)  # shield wire的吸引半径
                    # 矩阵的元素是同一数据类型，stroke_1第一列是高度，第二列是吸引半径;stroke_2是雷击点的位置名称-span_npa
                    stroke_1 = np.array([[yg, dg], [yc, dc]])  # sw;pc唯一1相
                    sc = self.Point_To_Line_Distance(C, D, P)  # 判断点到pc唯一1相的距离
                    stroke_2 = [["Direct", "shield wire", circles[i][0], sw_npa[i][0], 1],
                                ["Direct", "phase conductor", circlep[i][0,0], span_npa[i][0,0], conductor[0, 0]]]
                    # 谁高谁先判断-stroke_1 stroke_3 sall_points
                    indices = np.argsort(-stroke_1[:, 0])
                    stroke_1 = stroke_1[indices, :]
                    stroke_3 = np.array(stroke_2)[indices]
                    sall = np.array([ss, sc])  # 点到sw,pc唯一1相的距离
                    sall = sall[indices]
                    sall_points = np.array([[A, B], [C, D]])  # 跟哪条线段的sw或者pc有关系
                    sall_points = sall_points[indices]
                    if sall[0] <= stroke_1[0, 1]:
                        stroke_position.append(stroke_3[0, :])
                        stroke_point.append(sall_points[0, :])
                    elif sall[1] <= stroke_1[1, 1]:
                        stroke_position.append(stroke_3[1, :])
                        stroke_point.append(sall_points[1, :])
                    else:
                        stroke_position.append(np.array(["Indirect", "ground", np.nan, np.nan, np.nan]))
                        stroke_point.append(np.nan)


            # sw有2根
            else:
                A, B = span_min[i][:2]  # shield wire的第一根的起点终点
                U, V = span_min[i][3:5]  # shield wire的第二根的起点终点
                ss = np.zeros((1, 2))  # 判断点到shield wire的第i根距离
                ss[0, 0] = self.Point_To_Line_Distance(A, B, P)
                ss[0, 1] = self.Point_To_Line_Distance(U, V, P)
                # pc有2相
                if span_npa[i].size == 2:
                    C, D = span_pc[i][:2]  # pc的第1相的起点终点
                    E, F = span_pc[i][3:5]  # pc的第2相的起点终点
                    # 用公式计算吸引半径(sw与不同的pc比较会有不同的吸引半径)
                    da = np.zeros((2, 2))  # shield wire的第i根与pc的第i相距离差
                    da[0, 0] = np.sqrt((A[0] - C[0]) ** 2 + (A[1] - C[1]) ** 2)  # sw1pc1
                    da[0, 1] = np.sqrt((A[0] - E[0]) ** 2 + (A[1] - E[1]) ** 2)  # sw1pc2
                    da[1, 0] = np.sqrt((U[0] - C[0]) ** 2 + (U[1] - C[1]) ** 2)  # sw2pc1
                    da[1, 1] = np.sqrt((U[0] - E[0]) ** 2 + (U[1] - E[1]) ** 2)  # sw2pc2
                    yg = np.zeros((1, 2))  # shield wire的第i根的高度
                    yg[0, 0] = span_min[i][2]
                    yg[0, 1] = span_min[i][5]
                    yc = np.zeros((1, 2))  # pc的第i相的高度
                    yc[0, 0] = span_pc[i][2]
                    yc[0, 1] = span_pc[i][5]
                    # 用公式计算吸引半径
                    rc = 10 * I ** 0.65
                    rg = 5.5 * I ** 0.65
                    dc = np.zeros((2, 2))  # pc的吸引半径 %pc1对sw1,pc2对sw1; %pc1对sw2,pc2对sw2;
                    dg = np.zeros((2, 2))  # sw的吸引半径(对于第1相,对于第2相)%sw1对pc1,sw1对pc2;%sw2对pc1,sw2对pc2
                    alpha = np.zeros(2)
                    theta = np.zeros(2)
                    beta = np.zeros(2)
                    for swxks in range(2):
                        for pcxks in range(2):
                            alpha[pcxks] = np.arctan(
                                da[swxks, pcxks] / (yg[0, swxks] - yc[0, pcxks]))  # 单位是弧度,因为三角函数的输入参数默认为弧度
                            theta[pcxks] = np.arcsin((rg - yc[0, pcxks]) / rc)  # 单位是弧度
                            beta[pcxks] = np.arcsin(
                                ((yg[0, swxks] - yc[0, pcxks]) * np.sqrt(1 + np.tan(alpha[pcxks]) ** 2)) / (
                                            2 * rc))  # 单位是弧度
                            dc[swxks, pcxks] = rc * (np.cos(theta[pcxks]) - np.cos(alpha[pcxks] + beta[pcxks]))
                            dg[swxks, pcxks] = rc * np.cos(alpha[pcxks] - beta[pcxks])  # shield wire的吸引半径

                    sc = self.Point_To_Line_Distance(C, D, P)  # 判断点到pc的第1相的距离
                    sb = self.Point_To_Line_Distance(E, F, P)  # 判断点到pc的第2相的距离
                    stroke_1 = np.zeros((4, 2))  # 矩阵的元素是同一数据类型，stroke_1第一列是高度，第二列是吸引半径;
                    # 当点离pc的第i相更近，判断sw的吸引半径，先用sw对于第i相的吸引半径
                    if sc > sb and ss[0, 0] > ss[0, 1]:  # 点离pc的第2相更近且离sw的第2根更近
                        stroke_1[0, :] = np.array([yg[0, 0], dg[0, 1]])  # sw的第1根,吸引半径是sw1pc2
                        stroke_1[1, :] = np.array([yg[0, 1], dg[1, 1]])  # sw的第2根,吸引半径是sw2pc2
                        stroke_1[2, :] = np.array([yc[0, 0], dc[1, 0]])  # pc的第1相,吸引半径是pc1sw2
                        stroke_1[3, :] = np.array([yc[0, 1], dc[1, 1]])  # pc的第2相,吸引半径是pc2sw2
                    elif sc > sb and ss[0, 1] > ss[0, 0]:  # 点离pc的第2相更近且离sw的第1根更近
                        stroke_1[0, :] = np.array([yg[0, 0], dg[0, 1]])  # sw的第1根,吸引半径是sw1pc2
                        stroke_1[1, :] = np.array([yg[0, 1], dg[1, 1]])  # sw的第2根,吸引半径是sw2pc2
                        stroke_1[2, :] = np.array([yc[0, 0], dc[0, 0]])  # pc的第1相,吸引半径是pc1sw1
                        stroke_1[3, :] = np.array([yc[0, 1], dc[0, 1]])  # pc的第2相,吸引半径是pc2sw1
                    elif sb > sc and ss[0, 0] > ss[0, 1]:  # 点离pc的第1相更近且离sw的第2根更近
                        stroke_1[0, :] = np.array([yg[0, 0], dg[0, 0]])  # sw的第1根,吸引半径是sw1pc1
                        stroke_1[1, :] = np.array([yg[0, 1], dg[1, 0]])  # sw的第2根,吸引半径是sw2pc1
                        stroke_1[2, :] = np.array([yc[0, 0], dc[1, 0]])  # pc的第1相,吸引半径是pc1sw2
                        stroke_1[3, :] = np.array([yc[0, 1], dc[1, 1]])  # pc的第2相,吸引半径是pc2sw2
                    else:
                        stroke_1[0, :] = np.array([yg[0, 0], dg[0, 0]])  # sw的第1根,吸引半径是sw1pc1
                        stroke_1[1, :] = np.array([yg[0, 1], dg[1, 0]])  # sw的第2根,吸引半径是sw2pc1
                        stroke_1[2, :] = np.array([yc[0, 0], dc[0, 0]])  # pc的第1相,吸引半径是pc1sw1
                        stroke_1[3, :] = np.array([yc[0, 1], dc[0, 1]])  # pc的第2相,吸引半径是pc2sw1

                    # stroke_2是雷击点的位置名称-span_npa
                    stroke_2 = [["Direct", "shield wire", circles[i][0], sw_npa[i][0], 1],
                                ["Direct", "shield wire", circles[i][1], sw_npa[i][1], 2],  # sw的第1根,sw的第2根
                                ["Direct", "phase conductor", circlep[i][0, 0], span_npa[i][0, 0], conductor[0, 0]],
                                ["Direct", "phase conductor", circlep[i][0, 1], span_npa[i][0, 1], conductor[1, 0]]]  # pc的第1相,pc的第2相
                    # 谁高谁先判断-stroke_1 stroke_3 sall_points
                    indices = np.argsort(-stroke_1[:, 0])
                    stroke_1 = stroke_1[indices, :]
                    stroke_3 = np.array(stroke_2)[indices]
                    sall = np.array([ss[0, 0], ss[0, 1], sc, sb])  # 点到sw的第1根,sw的第2根,pc的第1相,pc的第2相的距离
                    sall = sall[indices]
                    sall_points = np.array([[A, B], [U, V], [C, D], [E, F]])  # 跟哪条线段的sw或者pc有关系
                    sall_points = sall_points[indices]
                    if sall[0] <= stroke_1[0, 1]:
                        stroke_position.append(stroke_3[0, :])
                        stroke_point.append(sall_points[0, :])
                    elif sall[1] <= stroke_1[1, 1]:
                        stroke_position.append(stroke_3[1, :])
                        stroke_point.append(sall_points[1, :])
                    elif sall[2] <= stroke_1[2, 1]:
                        stroke_position.append(stroke_3[2, :])
                        stroke_point.append(sall_points[2, :])
                    elif sall[3] <= stroke_1[3, 1]:
                        stroke_position.append(stroke_3[3, :])
                        stroke_point.append(sall_points[3, :])
                    else:
                        stroke_position.append(np.array(["Indirect", "ground", np.nan, np.nan, np.nan]))
                        stroke_point.append(np.nan)

                    # 当sw的2根高度一样时，判断两条sw的重叠区，离哪个sw近落在哪个sw
                    if stroke_position[-1][1] == "shield wire" and yg[0, 0] == yg[0, 1]:
                        # 点到sw的第1相的距离大于点到sw的第2相的距离
                        if ss[0, 0] > ss[0, 1]:
                            stroke_position[-1][2] = circles[i][1]
                            stroke_position[-1][4] = conductor[1, 0]
                            stroke_point[-1] = np.array([U, V])
                        else:
                            stroke_position[-1][2] = circles[i][0]
                            stroke_position[-1][4] = conductor[0, 0]
                            stroke_point[-1] = np.array([A, B])

                    # 当pc的2个相高度一样时，判断两条pc的重叠区，离哪个pc近落在哪个pc
                    if stroke_position[-1][1] == "phase conductor" and yc[0, 0] == yc[0, 1]:
                        # 点到pc的第1相的距离大于点到pc的第2相的距离
                        if sc > sb:
                            stroke_position[-1][2] = circlep[i][0, 1]
                            stroke_position[-1][3] = span_npa[i][0, 1]
                            stroke_position[-1][4] = conductor[1, 0]
                            stroke_point[-1] = np.array([E, F])
                        else:
                            stroke_position[-1][2] = circlep[i][0, 0]
                            stroke_position[-1][3] = span_npa[i][0, 0]
                            stroke_position[-1][4] = conductor[0, 0]
                            stroke_point[-1] = np.array([C, D])


                # pc有1相
                else:
                    C, D = span_pc[i][:2]  # pc唯一1相的起点终点
                    # 用公式计算吸引半径(sw与不同的pc比较会有不同的吸引半径)
                    da = np.zeros((1, 2))  # shield wire的第i根与pc唯一1相的距离差
                    da[0, 0] = np.sqrt((A[0] - C[0]) ** 2 + (A[1] - C[1]) ** 2)  # sw1pc
                    da[0, 1] = np.sqrt((U[0] - C[0]) ** 2 + (U[1] - C[1]) ** 2)  # sw2pc
                    yg = np.zeros((1, 2))  # shield wire的第i根的高度
                    yg[0, 0] = span_min[i][2]
                    yg[0, 1] = span_min[i][5]
                    yc = span_pc[i][2]  # pc的唯一1相的高度
                    # 用公式计算吸引半径
                    rc = 10 * I ** 0.65
                    rg = 5.5 * I ** 0.65
                    dc = np.zeros((1, 2))  # pc的吸引半径 %pc对sw1,pc对sw2
                    dg = np.zeros((1, 2))  # sw的吸引半径(对于第1相,对于第2相)%sw1对pc,sw2对pc
                    alpha = []
                    theta = []
                    beta = []
                    for swxks in range(2):
                        alpha = np.arctan(da[0, swxks] / (yg[0, swxks] - yc))  # 单位是弧度,因为三角函数的输入参数默认为弧度
                        theta = np.arcsin((rg - yc) / rc)  # 单位是弧度
                        beta = np.arcsin(((yg[0, swxks] - yc) * np.sqrt(1 + np.tan(alpha) ** 2)) / (2 * rc))  # 单位是弧度
                        dc[0, swxks] = rc * (np.cos(theta) - np.cos(alpha + beta))
                        dg[0, swxks] = rc * np.cos(alpha - beta)  # shield wire的吸引半径

                    sc = self.Point_To_Line_Distance(C, D, P)  # 判断点到pc唯一1相的距离
                    stroke_1 = np.zeros((3, 2))  # 矩阵的元素是同一数据类型，stroke_1第一列是高度，第二列是吸引半径;
                    # 当点离sw的第i相更近，判断pc的吸引半径，先用pc对于sw第i相的吸引半径
                    if ss[0, 0] > ss[0, 1]:  # 点离sw的第2根更近
                        stroke_1[0, :] = np.array([yg[0, 0], dg[0, 0]])  # sw的第1根,吸引半径是sw1pc
                        stroke_1[1, :] = np.array([yg[0, 1], dg[0, 1]])  # sw的第2根,吸引半径是sw2pc
                        stroke_1[2, :] = np.array([yc, dc[0, 1]])  # pc的唯一1相,吸引半径是pcsw2
                    else:  # 点离sw的第1根更近
                        stroke_1[0, :] = np.array([yg[0, 0], dg[0, 0]])  # sw的第1根,吸引半径是sw1pc
                        stroke_1[1, :] = np.array([yg[0, 1], dg[0, 1]])  # sw的第2根,吸引半径是sw2pc
                        stroke_1[2, :] = np.array([yc, dc[0, 0]])  # pc的唯一1相,吸引半径是pcsw1

                    # stroke_2是雷击点的位置名称-span_npa
                    stroke_2 = [["Direct", "shield wire", circles[i][0], sw_npa[i][0], 1],
                                ["Direct", "shield wire", circles[i][1], sw_npa[i][1], 2],  # sw的第1根,sw的第2根
                                ["Direct", "phase conductor", circlep[i][0, 0], span_npa[i][0, 0], conductor[0, 0]]]  # pc唯一1相
                    # 谁高谁先判断-stroke_1 stroke_3 sall_points
                    indices = np.argsort(-stroke_1[:, 0])
                    stroke_1 = stroke_1[indices, :]
                    stroke_3 = np.array(stroke_2)[indices]
                    sall = np.array([ss[0, 0], ss[0, 1], sc])  # 点到sw的第1根,sw的第2根,pc唯一1相的距离
                    sall = sall[indices]
                    sall_points = np.array([[A, B], [U, V], [C, D]])  # 跟哪条线段的sw或者pc有关系
                    sall_points = sall_points[indices]
                    if sall[0] <= stroke_1[0, 1]:
                        stroke_position.append(stroke_3[0, :])
                        stroke_point.append(sall_points[0, :])
                    elif sall[1] <= stroke_1[1, 1]:
                        stroke_position.append(stroke_3[1, :])
                        stroke_point.append(sall_points[1, :])
                    elif sall[2] <= stroke_1[2, 1]:
                        stroke_position.append(stroke_3[2, :])
                        stroke_point.append(sall_points[2, :])
                    else:
                        stroke_position.append(np.array(["Indirect", "ground", np.nan, np.nan, np.nan]))
                        stroke_point.append(np.nan)

                    # 当sw的2根高度一样时，判断两条sw的重叠区，离哪个sw近落在哪个sw
                    if stroke_position[-1][1] == "shield wire" and yg[0, 0] == yg[0, 1]:
                        # 点到sw的第1相的距离大于点到sw的第2相的距离
                        if ss[0, 0] > ss[0, 1]:
                            stroke_position[-1][2] = circles[i][1]
                            stroke_position[-1][4] = conductor[1, 0]
                            stroke_point[-1] = np.array([U, V])
                        else:
                            stroke_position[-1][2] = circles[i][0]
                            stroke_position[-1][4] = conductor[0, 0]
                            stroke_point[-1] = np.array([A, B])

        # stroke_position = np.array(stroke_position).T
        # stroke_point = np.array(stroke_point).T
        point_medium = [np.nan] * len(points_need)
        segment_position = np.zeros(len(points_need))
        dmin_allID = np.nan * np.ones(len(points_need))
        segmentID = np.nan * np.ones(len(points_need))

        # 判断直接雷的落雷点是Tower/Span-segment_position,落雷点坐标-point_medium
        # 把有关系的这条线段的sw或者pc分为几等分，判断点落在几等分的哪个等分段-segmentID
        for i in range(points_need.shape[0]):
            if stroke_position[i][0] == "Direct":
                P = points_need[i, :]  # 待判断的落雷点
                line_segment = stroke_point[i]
                A, B = line_segment[0, :], line_segment[1, :]
                AP = P - A
                AB = B - A
                # 判断点落在segments等分的哪个等分段
                # 点P到线段AB的垂直交点G
                t = np.dot(AP, AB) / np.dot(AB, AB)
                G = A + t * AB
                stroke_point[i] = np.array([G[0], G[1]])
                # 落雷点会被吸引落在（segments+1）个点中离得最近的点
                ABsegment = esegment[i]
                if ABsegment == 1:
                    # 计算点 P 与端点A,B的距离
                    point_text = np.array([A, B])
                    distances = np.sqrt((point_text[:, 0] - G[0]) ** 2 + (point_text[:, 1] - G[1]) ** 2)
                    # 找到距离最近的点的坐标-nearest_point
                    segment_position[i] = 1
                    segmentID[i] = 1
                    min_distance, min_index = distances.min(), distances.argmin()
                    if min_index == 0:
                        dmin_allID[i] = Edges[dmin_all[i], 0]
                    else:
                        dmin_allID[i] = Edges[dmin_all[i], 1]

                    nearest_point = point_text[min_index, :]
                    point_medium[i] = nearest_point
                else:
                    TAB = np.linalg.norm(AB)  # 计算总长度
                    PerTAB = TAB / ABsegment  # 每段的长度
                    # 初始化存储点的数组
                    segments_points = np.zeros(((int(ABsegment) - 1), 2))
                    # 计算每段的点
                    for ise in range((int(ABsegment) - 1)):
                        t = (ise + 1) / ABsegment
                        segments_points[ise, 0] = A[0] + t * (B[0] - A[0])
                        segments_points[ise, 1] = A[1] + t * (B[1] - A[1])

                    # 计算点 P 与所有点的距离
                    point_text = np.vstack([A, B, segments_points])
                    distances = np.sqrt((point_text[:, 0] - G[0]) ** 2 + (point_text[:, 1] - G[1]) ** 2)
                    # 找到距离最近的点的坐标-nearest_point
                    min_distance, min_index = distances.min(), distances.argmin()
                    nearest_point = point_text[min_index, :]
                    # 两端是Tower,中间的点是Span
                    if min_index == 0:
                        segment_position[i] = 1
                        dmin_allID[i] = Edges[dmin_all[i], 0]
                    elif min_index == 1:
                        segment_position[i] = 1
                        dmin_allID[i] = Edges[dmin_all[i], 1]
                    else:
                        segment_position[i] = 2
                        dmin_allID[i] = dmin_all[i]

                    point_medium[i] = nearest_point
                    point_text2 = np.vstack([A, segments_points, B])
                    point_text2_x = point_text2[:, 0]
                    seg_id = np.where((G[0] >= point_text2_x[:-1]) & (G[0] <= point_text2_x[1:]))[0]
                    if seg_id.size > 0:
                        segmentID[i] = seg_id[0]
                    else:
                        if nearest_point.all() == A.all():
                            segmentID[i] = 1
                        else:
                            segmentID[i] = ABsegment

        # stroke_position是判断直接雷(Dierect)/间接雷(Indirect),（sw/pc）/（ground）,sw/pc的circle_ID,pc对应#相的phase_ID,conductor_ID;
        # stroke_point是落雷点的最终位置（ground(间接雷)-坐标不变,显示NaN，sw/pc(直接雷)-点到线段的垂直交点;
        # segment_position2是雷的落雷点是Tower(1)/Span(2)/ground(0)
        # point_medium是落在有关线段（sw/pc）的3等分的哪个等分点
        # dmin_allID2是Tower/Span的ID
        points_need2 = [list(row) for row in points_need]
        stroke_position3 = np.stack(stroke_position)
        stroke_position21 = [stroke_position3[:, i] for i in range(stroke_position3.shape[1])]
        stroke_position2 = [list(arr) for arr in stroke_position21]
        dmin_allID2 = dmin_allID + 1  # python是0-based，实际情况的序号是从1开始
        segmentID2 = segmentID + 1  # python是0-based，实际情况的序号是从1开始
        stroke_result = [list(a) for a in zip(points_need2, stroke_position2[0], stroke_position2[1],
                                              stroke_position2[2], stroke_position2[3],  stroke_position2[4], stroke_point,
                                              segment_position, point_medium, dmin_allID2, segmentID2)]
        stroke_result = [list(item) for item in zip(*stroke_result)]
        # 排列顺序
        stroke_result2 = stroke_result[:-1]
        stroke_result2[0] = stroke_result[1]
        stroke_result2[1] = stroke_result[7]
        stroke_result2[2] = stroke_result[9]
        stroke_result2[3] = stroke_result[3]
        stroke_result2[4] = stroke_result[4]
        stroke_result2[5] = stroke_result[5]
        stroke_result2[6] = stroke_result[10]
        stroke_result2[7] = stroke_result[0]
        stroke_result2[8] = stroke_result[6]
        stroke_result2[9] = stroke_result[8]


        # 同一个flash的每一次stroke的位置跟第一个stroke是一样的
        wp = 0
        while wp < len(siteone):
            esite = siteone[wp]
            if wp != len(siteone) - 1:
                esitenext = siteone[wp + 1]
                for sc in range(len(stroke_result2)):
                    for sr in range(esite, esitenext):
                        if sc != 7:
                            stroke_result2[sc][sr] = stroke_result2[sc][esite]

            else:
                esitenext = len(stroke_result2[0]) - 1
                for sc in range(len(stroke_result2)):
                    for sr in range(esite, esitenext + 1):
                        if sc != 7:
                            stroke_result2[sc][sr] = stroke_result2[sc][esite]

            wp += 1

        # 落雷点所有要求的数据-stroke_result
        ## Direct/Indirect,Tower(1)/Span(2)/ground(0),Tower/Span的ID,sw/pc的circle_ID
        ## pc对应#相的phase_ID,conductor_ID,落在第几个等分段，落雷点的XY坐标，落雷点的最终位置
        stroke_result = stroke_result2[0:9]  # 变量-次数
        stroke_result5 = [list(item) for item in zip(*stroke_result)]  # 次数-变量

        ## 直接雷，间接雷的index
        sited = [index for index, value in enumerate(stroke_result[0]) if value == 'Direct']
        stroke_d = [stroke_result5[i] for i in sited]
        siteind = [index for index, value in enumerate(stroke_result[0]) if value == 'Indirect']
        stroke_ind = [stroke_result5[i] for i in siteind]
        sitesw = [index for index, value in enumerate(stroke_result[4]) if value == '0.0']
        sitepc = [index for index, value in enumerate(stroke_result[4]) if value in ['1.0', '2.0', '3.0']]
        sitet = [index for index, value in enumerate(stroke_result[1]) if value == 1.0]
        sitesp = [index for index, value in enumerate(stroke_result[1]) if value == 2.0]
        # 直接雷（sw/pc）中shield wire，phase conductor的占比
        count_d = len(sited)  # 直接雷次数
        count_ind = len(siteind)  # 间接雷次数
        count_sw = len(sitesw)  # shield wire次数
        count_pc = len(sitepc)  # phase conductor次数
        count_t = len(sitet)  # Tower次数
        count_sp = len(sitesp)  # Span次数
        pdd = count_d / (count_d + count_ind)
        pind = count_ind / (count_d + count_ind)
        if count_sw + count_pc == 0:
            psw = ppc = np.nan
        else:
            psw = count_sw / (count_sw + count_pc)
            ppc = count_pc / (count_sw + count_pc)

        if count_t + count_sp == 0:
            pt = psp = np.nan
        else:
            pt = count_t / (count_t + count_sp)
            psp = count_sp / (count_t + count_sp)

        # 直接雷的heatmap-point_medium
        point_medium = stroke_result2[9]
        # 初始化一个计数器
        count_map = {}
        # 遍历元胞数组，计算每个点出现的次数
        for point in point_medium:
            if point is not None and not np.isnan(point).any():
                key = str(point)
                if key in count_map:
                    count_map[key] += 1
                else:
                    count_map[key] = 1

        # 提取坐标和出现次数
        coordinates = []
        counts = []
        for key, value in count_map.items():
            # 使用正则表达式从字符串中提取数字
            nums = re.findall(r'[-+]?\d*\.?\d+(?:e[-+]?\d+)?', key)
            # 将提取的字符串数字转换为浮点数，并将结果作为列表添加到 coordinates 中
            coordinates.append([float(num) for num in nums])
            # 将出现次数添加到 counts 中
            counts.append(value)
        # 提取 x 和 y 坐标
        x_coordinates = [coord[0] for coord in coordinates]
        y_coordinates = [coord[1] for coord in coordinates]
        # 创建颜色映射矩阵，根据 c 的值选择对应的颜色
        # 归一化 counts 到 [0, 1] 范围
        min_count = min(counts)
        max_count = max(counts) + 1e-7
        c = [(count - min_count) / (max_count - min_count) for count in counts]
        # 创建散点图
        plt.figure(4)
        plt.scatter(points_need[:, 0], points_need[:, 1], 1, 'k', label='points_need')
        scatter = plt.scatter(x_coordinates, y_coordinates, 50, c, cmap='jet', label='point_medium', alpha=0.6)
        plt.colorbar(scatter)  # 显示颜色条
        plt.xlabel('Horizontal distance (m)')
        plt.ylabel('Vertical distance (m)')
        plt.title(f'Maximum number of lightning strikes: {max(counts)}')  # 标题解释最多次数的直接雷的落雷点的次数，即1代表多少次数
        plt.show()

        # # 输出分类次数的txt格式
        # # 构建foldname文件的路径
        # filepath = os.path.join(foldname, 'Statistics Summary.txt')
        # # 打开文件并写入
        # with open(filepath, 'w') as file:
        #     file.write(f"The total number of flash is: {len(unique_flash)}\n")
        #     file.write(f"The total number of stroke is: {stroke_counts.sum()}\n")
        #     file.write(f"The area of surrounding lines is: {polygonArea}\n")
        #     file.write(f"The number of points within the bounding line: {len(points_need)}\n")
        #     file.write(f"The number of Direct light: {count_d}\n")
        #     file.write(f"The number of Indirect light: {count_ind}\n")
        #     file.write(f"The proportion of Direct light is: {pdd}\n")
        #     file.write(f"The proportion of Indirect light is: {pind}\n")
        #     file.write(f"The number of shield wire: {count_sw}\n")
        #     file.write(f"The number of phase conductor: {count_pc}\n")
        #     file.write(f"The proportion of shield wire is: {psw}\n")
        #     file.write(f"The proportion of phase conductor is: {ppc}\n")
        #     file.write(f"The number of Tower-Direct light: {count_t}\n")
        #     file.write(f"The number of Span-Direct light: {count_sp}\n")
        #     file.write(f"The proportion of Tower-Direct light is: {pt}\n")
        #     file.write(f"The proportion of Span-Direct light is: {psp}\n")

        # 输出分类次数的excel格式
        dataSTS = [
            ['The total number of flash is', len(unique_flash)],
            ['The total number of stroke is', stroke_counts.sum()],
            ['The area of surrounding lines is', polygonArea],
            ['The number of points within the bounding line', len(points_need)],
            ['The number of Direct light', count_d],
            ['The number of Indirect light', count_ind],
            ['The proportion of Direct light is', pdd],
            ['The proportion of Indirect light is', pind],
            ['The number of shield wire', count_sw],
            ['The number of phase conductor', count_pc],
            ['The proportion of shield wire is', psw],
            ['The proportion of phase conductor is', ppc],
            ['The number of Tower-Direct light', count_t],
            ['The number of Span-Direct light', count_sp],
            ['The proportion of Tower-Direct light is', pt],
            ['The proportion of Span-Direct light is', psp],
        ]
        # 将double 数组转换为table
        STS = pd.DataFrame(dataSTS, columns=['Description', 'Value'])
        # 将文件夹名添加到文件路径
        outputFile = os.path.join(foldname, 'Statistics Summary.xlsx')
        #检查文件是否存在并删除
        try:
            os.remove(outputFile)  # 如果文件存在，删除它
        except FileNotFoundError:
            pass

        # 将 DataFrame 保存为 Excel 文件
        STS.to_excel(outputFile, index=False)

        # save
        if userInput3 == 1:
            # 将double 数组转换为table
            df3 = pd.DataFrame(stroke_result5, columns=['Direct_or_Indirect', 'Tower1_or_Span2_or_Ground0',
                                                        'Tower_or_Span_ID', 'Circle_ID', 'phase_ID','conductor_ID', 'segment_ID',
                                                        'coordinates_of_point_XY',
                                                        'Location_of_the_lightning_strike_point_XY'])
            # 将文件夹名添加到文件路径
            outputFile = os.path.join(foldname, 'Stroke Position.xlsx')
            # 检查文件是否存在并删除
            try:
                os.remove(outputFile)  # 如果文件存在，删除它
            except FileNotFoundError:
                pass

            # 将 DataFrame 保存为 Excel 文件
            df3.to_excel(outputFile, index=False)

        if userInput3d == 1:
            # 将double 数组转换为table
            df3d = pd.DataFrame(stroke_d, columns=['Direct_or_Indirect', 'Tower1_or_Span2_or_Ground0',
                                                   'Tower_or_Span_ID', 'Circle_ID', 'phase_ID','conductor_ID', 'segment_ID',
                                                   'coordinates_of_point_XY',
                                                   'Location_of_the_lightning_strike_point_XY'])
            # 将文件夹名添加到文件路径
            outputFile = os.path.join(foldname, 'DIR Stroke Position.xlsx')
            # 检查文件是否存在并删除
            try:
                os.remove(outputFile)  # 如果文件存在，删除它
            except FileNotFoundError:
                pass

            # 将 DataFrame 保存为 Excel 文件
            df3d.to_excel(outputFile, index=False)

        if userInput3ind == 1:
            # 将double 数组转换为table
            df3ind = pd.DataFrame(stroke_ind, columns=['Direct_or_Indirect', 'Tower1_or_Span2_or_Ground0',
                                                       'Tower_or_Span_ID', 'Circle_ID', 'phase_ID', 'conductor_ID','segment_ID',
                                                       'coordinates_of_point_XY',
                                                       'Location_of_the_lightning_strike_point_XY'])
            # 将文件夹名添加到文件路径
            outputFile = os.path.join(foldname, 'IND Stroke Position.xlsx')
            # 检查文件是否存在并删除
            try:
                os.remove(outputFile)  # 如果文件存在，删除它
            except FileNotFoundError:
                pass

            # 将 DataFrame 保存为 Excel 文件
            df3ind.to_excel(outputFile, index=False)

        # 所有统计参数写成struct结构，用一个变量传递数据
        resultstro = {'stroke_result': stroke_result,
                      'sited': sited,
                      'siteind': siteind,
                      'sitesw': sitesw,
                      'sitepc': sitepc,
                      'sitet': sitet,
                      'sitesp': sitesp,
                      'pdd': pdd,
                      'pind': pind,
                      'psw': psw,
                      'ppc': ppc,
                      'pt': pt,
                      'psp': psp}

        return resultstro,dataSTS,stroke_result

    def Current_Waveform_Generator(self, Wave_Model, DSave, resultstro, resultcur, foldname):
        # struct获取每个变量名
        way = Wave_Model
        userInput4 = DSave['userInput4']
        userInput4d = DSave['userInput4d']
        userInput4ind = DSave['userInput4ind']
        sited = [int(x) for x in resultstro['sited']]
        siteind = resultstro['siteind']
        parameterst = resultcur['parameterst']

        if way == 1:
            light_final = self.WaveformM(parameterst)

        if way == 2:
            light_final = self.WaveformG(parameterst)

        # save
        if userInput4 == 1 and way == 1:
            # 将double 数组转换为table
            df4 = pd.DataFrame(light_final, columns=['tn', 'A', 'B', 'n',
                                                     'I1', 't1', 'I2', 't2', 'Ipi', 'Ipc'])
            # 将文件夹名添加到文件路径
            outputFile = os.path.join(foldname, 'Current Waveform_CIGRE.xlsx')
            # 检查文件是否存在并删除
            try:
                os.remove(outputFile)  # 如果文件存在，删除它
            except FileNotFoundError:
                pass

            # 将 DataFrame 保存为 Excel 文件
            df4.to_excel(outputFile, index=False)
            filepath = self.path + '/' + 'DATA_MCLG/Cigre Function.txt'
            # filepath = "H:\Ph.D. Python\StrongEMPPlatform\Tower\Predecessor\DATA_MCLG/b_Cigre Function.txt"
            # 打开文件并写入
            with open(filepath, 'w') as file:
                file.write(f"syms t\n")
                file.write(
                    f"i(t)=((t<= tn).* (A*t+B*(t^n)) + (t>tn) * (I1*exp(-(t-tn)/t1) - I2*exp(-(t-tn)/t2)))*(Ipi/Ipc)\n")

        elif userInput4 == 1 and way == 2:
            df4 = pd.DataFrame(light_final, columns=['I0', 'nm', 't1', 'N', 't2'])
            # 将文件夹名添加到文件路径
            outputFile = os.path.join(foldname, 'Current Waveform_HEIDLER.xlsx')
            # 检查文件是否存在并删除
            try:
                os.remove(outputFile)  # 如果文件存在，删除它
            except FileNotFoundError:
                pass

            # 将 DataFrame 保存为 Excel 文件
            df4.to_excel(outputFile, index=False)
            filepath =self.path + '/' + 'DATA_MCLG/Heidler Function.txt'
            # 打开文件并写入
            with open(filepath, 'w') as file:
                file.write(f"syms t\n")
                file.write(f"i(t)=(I0/nm)*(((t/t1)^N)/(1+(t/t1)^N))*exp(-t/t2)\n")

        if userInput4d == 1:  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            df4d = pd.DataFrame(np.array(light_final)[sited,:], columns=['tn', 'A', 'B', 'n',
                                                                'I1', 't1', 'I2', 't2', 'Ipi', 'Ipc'])
            # 将文件夹名添加到文件路径
            outputFile = os.path.join(foldname, 'DIR Current Waveform CIGRE.xlsx')
            # 检查文件是否存在并删除
            try:
                os.remove(outputFile)  # 如果文件存在，删除它
            except FileNotFoundError:
                pass

            # 将 DataFrame 保存为 Excel 文件
            df4d.to_excel(outputFile, index=False)

        if userInput4ind == 1:  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            df4ind = pd.DataFrame(np.array(light_final)[siteind, :], columns=['tn', 'A', 'B', 'n',
                                                                    'I1', 't1', 'I2', 't2', 'Ipi', 'Ipc'])
            # 将文件夹名添加到文件路径
            outputFile = os.path.join(foldname, 'IND Current Waveform CIGRE.xlsx')
            # 检查文件是否存在并删除
            try:
                os.remove(outputFile)  # 如果文件存在，删除它
            except FileNotFoundError:
                pass

            # 将 DataFrame 保存为 Excel 文件
            df4ind.to_excel(outputFile, index=False)

        # 所有统计参数写成struct结构，用一个变量传递数据
        resultwave = {'light_final': light_final}

        return resultwave
    def Point_To_Line_Distance(self, A, B, P):
        AB = B - A
        AP = P - A
        BP = P - B
        # Scalar projection
        t = np.dot(AP, AB) / np.dot(AB, AB)
        if t < 0:
            # Closest point is A
            d = np.linalg.norm(AP)
        elif t > 1:
            # Closest point is B
            d = np.linalg.norm(BP)
        else:
            # Closest point is on the segment
            d = np.linalg.norm(AP - t * AB)

        return d
    def WaveformM(self, parameterst):
        light_final = []
        with ProcessPoolExecutor() as executor:
            results = self.Compute_Light_Final(parameterst)
            for result in results:
                light_final.append(result)
        return light_final

    def Compute_Light_Final(self, params):
        light_final = []
        a = params.shape[0]
        for i in range(0, params.shape[0] + 1):
        # for each quadruple of values for Ip , tf and th
        # 换算单位,实际Ipi-kA-A，tfi-μs-s，Smi-kA/μs-A/s，thi-μs-s
            Ipi, tfi, Smi, thi = params[i - 1, 2] * 1e3, params[i - 1, 3] * 1e-6, params[i - 1, 4] * (1e3 / 1e-6), params[i - 1, 5] * 1e-6

            SN = Smi * tfi / Ipi
            n = 1 + 2 * (SN - 1) * (2 + 1 / SN)
            # In case a Monte Carlo event presents a value of n out of these bounds, the value of Sm is adjusted as
            if n < 1:
                Smi = 1.01 * Ipi / tfi
                SN = Smi * tfi / Ipi
                n = 1 + 2 * (SN - 1) * (2 + 1 / SN)
            elif n > 55:
                Smi = 12 * Ipi / tfi
                SN = Smi * tfi / Ipi
                n = 1 + 2 * (SN - 1) * (2 + 1 / SN)

            tn = 0.6 * tfi * (3 * SN ** 2 / (1 + SN ** 2))
            A = (1 / (n - 1)) * (0.9 * (Ipi / tn) * n - Smi)
            B = (1 / ((tn ** n) * (n - 1))) * (Smi * tn - 0.9 * Ipi)
            t1, t2 = (thi - tn) / np.log(2), 0.1 * Ipi / Smi
            # 按照文章编写公式错误，细心
            I1 = ((t1 * t2) / (t1 - t2)) * (Smi + 0.9 * (Ipi / t2))
            I2 = ((t1 * t2) / (t1 - t2)) * (Smi + 0.9 * (Ipi / t1))

            # 求解最大值Ipc(Ipc对应的时间的范围是[0,50]),单位是μs!!!
            # 缩小搜索范围，使速度更快
            lb, ub = (tn / 10 ** (np.log10(tn)) * 10 - 5) * (10 ** (np.log10(tn) - 1)), \
                     (tn / 10 ** (np.log10(tn)) * 10 + 5) * (10 ** (np.log10(tn) - 1))
            # 第2种方法：用时4.734314 秒，精度与第1种方法差不多
            step = 10 ** (np.log10(lb) - 3)
            num = int((ub - lb) / step) + 1
            t = np.linspace(lb, ub, num)
            y = np.where(t <= tn, A * t + B * (t ** n), I1 * np.exp(-(t - tn) / t1) - I2 * np.exp(-(t - tn) / t2))
            # 找到 y 中的最大值及其对应的索引
            Ipc = np.max(y)

            # As this procedure can lead to small errors on the resulting
            # current peak, the current is normalized to the desired peak value.
            # syms t
            # Cigre Function
            y2 = lambda t: ((t <= tn) * (A * t + B * (t ** n)) + (t > tn) * (
                    I1 * np.exp(-(t - tn) / t1) - I2 * np.exp(-(t - tn) / t2))) * (Ipi / Ipc)

            # lighting parameters:tn A B n I1 t1 I2 t2 Ipi Ipc

            new_row = [tn, A, B, n, I1, t1, I2, t2, Ipi, Ipc]
            light_final.append(new_row)
        return light_final

    def WaveformG(self, parameterst):
        lightp = parameterst[:, 2:]
        light_final = np.zeros((len(parameterst), 5))

        for i in range(len(lightp)):
            #  for each quadruple of values for Ip , tf and th
            Ipi, tfi, thi = lightp[i, 0], lightp[i, 1], lightp[i, 3]
            # At first the values of c1, c2 and c3 are equal to each other.
            c1 = c2 = c3 = 1
            # 上下限的范围是[I0，t1,t2,N,Ip,tf,th]
            # I0是初始状态下的电流（Ip的范围是[3,200]）；t1，t2是信号传输线路中的时延或响应时间（tf的范围是[0.1,30] and
            # th的范围是[1,500]）
            bounds = [(1, 200), (0.1, 30), (1, 500), (2, 4), (3, 200), (0.1, 30), (1, 500)]
            tstart = time()
            # %画图 %并行计算,是否并行%显示每次迭代过程
            # The initial population size =50, The maximum number of generations= 100,
            # 遗传算法，未知数是I0，t1,t2,N,Ip,tf,th
            result = differential_evolution(
                self.Objhei(),
                bounds,
                args=(c1, c2, c3, Ipi, tfi, thi, tstart),
                strategy='best1bin',
                maxiter=100,
                popsize=50,
                tol=0.01,
                mutation=(0.5, 1),
                recombination=0.7,
                seed=None,
                callback=None,
                disp=True,
                polish=True,
                init='latinhypercube',
                atol=0
            )

            # 输出最终解
            x = result.x
            # The possible values of N are limited to the integer values 2, 3 or 4.
            I0, t1, t2, N = x[0], x[1], x[2], round(x[3])
            nm = np.exp((-t1 / t2) * (t2 * N / t1) ** (1 / N))
            Ipc, tfc, thc = x[4], x[5], x[6]

            light_final[i, :] = [I0, nm, t1, N, t2]

        return light_final

    def Objhei(self, x, c1, c2, c3, Ipi, tfi, thi, tstart):
        Ipc, tfc, thc = self.Heidler(x)
        I0 = x[0]
        t1 = x[1]
        t2 = x[2]
        N = round(x[3])
        x[4] = Ipc
        x[5] = tfc
        x[6] = thc

        # If after some tens of attempts, conditions (13) are still not satisfied,
        # then the time to half value is penalized by means of a reduction of c3 with respect to c1 and c2.
        # 假设c3是随着运行次数的增加而逐渐减小,而不是一次性减小并在后续尝试中保持不变
        if time() - tstart >= 5 * 60:
            c3 = c3 * 0.9 ** ((time() - tstart) / (5 * 60))

        # The algorithm is stopped if the relative errors on the three parameters satisfy all the three following conditions:
        # the conditions of (13)
        if abs((Ipc - Ipi) / Ipi) < 0.5e-2 and abs((tfc - tfi) / tfi) < 0.5e-2 and abs((thc - thi) / thi) < 1e-2:
            f = -1e10
        else:
            f = c1 * abs((Ipc - Ipi) / Ipi) + c2 * abs((tfc - tfi) / tfi) + c3 * abs((thc - thi) / thi)

        return f

    def Heidler(self, x):
        # Heidler Function
        # The possible values of N are limited to the integer values 2, 3 or 4.
        I0, t1, t2 = x[0], x[1], x[2]
        N = round(x[3])
        nm = exp((-t1 / t2) * (t2 * N / t1) ** (1 / N))
        # 将符号表达式转换为可供 minimize_scalar 使用的函数
        im2 = lambda tm: -((I0 / nm) * (((tm / t1) ** N) / (1 + (tm / t1) ** N)) * exp(-tm / t2).evalf())
        tm = symbols('tm')
        im = (I0 / nm) * (((tm / t1) ** N) / (1 + (tm / t1) ** N)) * exp(-tm / t2)

        # Ipc(Ipc对应的时间的范围是[0,50])!!!!!!!!!!!!
        res = minimize_scalar(im2, bounds=(0, 50), method='bounded')
        Ipc = abs(res.fun)  # 搜索过程中可能会出现复数
        # 搜索过程中可能会出现Ipc = []或Ipc=NaN
        if len(Ipc) == 0 or np.isnan(Ipc):
            Ipc = -100

        # tfc:time from 第一次10%*Ipc到第一次90%*Ipc
        tfc1_sol = solve(im - 0.1 * Ipc, tm)
        tfc2_sol = solve(im - 0.9 * Ipc, tm)
        tfc1 = np.min([float(sol) for sol in tfc1_sol])
        tfc2 = np.min([float(sol) for sol in tfc2_sol])
        tfc = abs(tfc2 - tfc1)
        # 搜索过程中可能会出现tfc = []
        if np.isnan(Ipc):
            tfc = -100

        #  thc：time from 第一次10%*Ipc到第二次50%*Ipc
        thc2_sol = solve(im - 0.5 * Ipc, tm)
        thc2 = np.max([float(sol) for sol in thc2_sol])
        thc = abs(thc2 - tfc1)
        # 搜索过程中可能会出现thc = []
        if np.isnan(Ipc):
            thc = -100

        return Ipc, tfc, thc

