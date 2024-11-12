import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import re

from Tower_V3.PARA_MCLG.OTHS.point_to_line_distance import point_to_line_distance

def Lightning_stroke_location(Line, resultcur, DSave, foldname, resultedge,AR):
    # struct获取每个变量名
    Coordinates = Line['Node']
    Edges = Line['Edges'].astype(int)
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
    Phase1 = AR['Phase1']
    Phase2 = AR['Phase2']
    slopep = AR['slopep']
    buildingp = AR['buildingp']


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
    slopestart = [] #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    esegment = np.zeros(points_need.shape[0]).astype(int)
    dmin_all = np.zeros(points_need.shape[0])  # 每一个落雷点最近的边的编号
    for tp in range(points_need.shape[0]):
        P = points_need[tp, :]  # 待判断的落雷点
        d = np.zeros(len(origin_line))
        # 每一条线段
        for to in range(len(origin_line)):
            A, B = origin_line[to][0], origin_line[to][1]  # 线段的端点
            d[to] = point_to_line_distance (A, B, P)

        dmin_index = np.argmin(d)
        dmin_all[tp] = dmin_index
        # segments = np.array(segments[0, dmin_index])
        esegment[tp] = segments[0, dmin_index]
        span0 = OHLPf[dmin_index][0]
        slopestart.append([span0[0,0:2], span0[0, 3:5]]) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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
        # a = e_circle[ev_span_np, 1]
        ev_span_npa = e_circle[ev_span_np, 1].T
        ev_sw_npa = e_circle[1:(SWnumber[0, dmin_all[fh]] + 1), 1].T
        ecirclep = e_circle[ev_span_np, 0].T
        ecircles = e_circle[1:(SWnumber[0, dmin_all[fh]] + 1), 0].T

        span_npa.append(ev_span_npa)
        sw_npa.append(ev_sw_npa)
        circlep.append(ecirclep)
        circles.append(ecircles)

    conductor = np.array(conductor[0])
    # EGM model
    # 判断每个点points_need雷击点的位置-stroke_result
    stroke_position = []
    stroke_height = [] #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    stroke_r = []  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    stroke_distance = []  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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
                rc = Phase1[0] * I ** Phase1[1]
                dc = [rc, rc]
                # 矩阵的元素是同一数据类型，stroke_1第一列是高度，第二列是吸引半径;stroke_2是雷击点的位置名称-span_npa
                stroke_1 = np.array([[yc[0, 0], dc[0], np.nan], [yc[0, 1], dc[1], np.nan]])  # pc的第1相;pc的第2相
                sc = point_to_line_distance(C, D, P)  # 判断点到pc的第1相的距离
                sb = point_to_line_distance(E, F, P)  # 判断点到pc的第2相的距离
                stroke_2 = [["Direct", "phase conductor", circlep[i][0, 0], span_npa[i][0, 0], conductor[0, 0]],
                            ["Direct", "phase conductor", circlep[i][0, 1], span_npa[i][0, 1], conductor[1, 0]]]
                # 谁高谁先判断-stroke_1 stroke_3 sall_points
                indices = np.argsort(-stroke_1[:, 0])
                stroke_1 = stroke_1[indices, :]
                stroke_3 = np.array(stroke_2)[indices]
                sall = np.array([sc, sb])  # 点到pc的第1相,pc的第2相的距离
                sall = sall[indices]
                sall_points = np.array([[C, D], [E, F]])  # 跟哪条线段的pc有关系
                sall_points = sall_points[indices]
                if sall[0] <= stroke_1[0, 1]:
                    stroke_height.append(stroke_1[0, 0]) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    stroke_r.append(stroke_1[0, 1]) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    stroke_distance.append(sall[0])  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    stroke_position.append(stroke_3[0, :])
                    stroke_point.append(sall_points[0, :])
                elif sall[1] <= stroke_1[1, 1]:
                    stroke_height.append(stroke_1[1, 0])  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    stroke_r.append(stroke_1[1, 1])  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    stroke_distance.append(sall[1])  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    stroke_position.append(stroke_3[1, :])
                    stroke_point.append(sall_points[1, :])
                else:
                    stroke_height.append(0)  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    stroke_r.append(np.nan)  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    stroke_distance.append(np.nan)  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    stroke_position.append(np.array(["Indirect", "ground", np.nan, np.nan, np.nan]))
                    stroke_point.append([np.nan, np.nan])

                # 当pc的2个相高度一样时，判断两条pc的重叠区，离哪个pc近落在哪个pc
                if stroke_position[-1][1] == "phase conductor" and yc[0, 0] == yc[0, 1]:
                    # 点到pc的第1相的距离大于点到pc的第2相的距离
                    if sc > sb:
                        stroke_distance.append(sb)  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        stroke_position[-1][2] = circlep[i][0, 1]
                        stroke_position[-1][3] = span_npa[i][0, 1]
                        stroke_position[-1][4] = conductor[1, 0]
                        stroke_point[-1] = np.array([E, F])
                    else:
                        stroke_distance.append(sc)  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        stroke_position[-1][2] = circlep[i][0, 0]
                        stroke_position[-1][3] = span_npa[i][0, 0]
                        stroke_position[-1][4] = conductor[0, 0]
                        stroke_point[-1] = np.array([C, D])

            # pc有1相
            else:
                C, D = span_pc[i][:2]  # pc唯一1相的起点终点
                stroke_height.append(span_pc[i][2])  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # 用公式计算吸引半径
                rc = Phase1[0] * I ** Phase1[1]
                dc = rc  # pc的吸引半径
                sc = point_to_line_distance(C, D, P)  # 判断点到pc唯一1相的距离
                stroke_r.append(dc)  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                if sc <= dc:
                    stroke_distance.append(sc)  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    stroke_position.append(["Direct", "phase conductor", circlep[i][0, 0], span_npa[i][0, 0]],
                                           conductor[0, 0])
                    stroke_point.append(np.array([C, D]))
                else:
                    stroke_distance.append(np.nan)  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    stroke_position.append(np.array(["Indirect", "ground", np.nan, np.nan, np.nan]))
                    stroke_point.append([np.nan, np.nan])

        # sw有1根
        elif sum(x is not None for x in span_min[i]) == 3:
            A, B = span_min[i][:2]  # shield wire的起点终点
            ss = point_to_line_distance(A, B, P)  # 判断点到shield wire的距离
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
                rc = Phase1[0] * I ** Phase1[1]
                rg = Phase2[0] * I ** Phase2[1]
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
                sc = point_to_line_distance(C, D, P)  # 判断点到pc的第1相的距离
                sb = point_to_line_distance(E, F, P)  # 判断点到pc的第1相的距离
                # 当点离pc的第i相更近，判断sw的吸引半径，先用sw对于第i相的吸引半径
                if sc > sb:
                    stroke_1[0, :] = np.array([yg, dg[0, 1], np.nan])
                else:
                    stroke_1[0, :] = np.array([yg, dg[0, 0], np.nan])

                stroke_2 = [["Direct", "shield wire", circles[i][0], sw_npa[i][0], 1],
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
                    stroke_height.append(stroke_1[0, 0])  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    stroke_r.append(stroke_1[0, 1])  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    stroke_distance.append(sall[0])  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    stroke_position.append(stroke_3[0, :])
                    stroke_point.append(sall_points[0, :])
                elif sall[1] <= stroke_1[1, 1]:
                    stroke_height.append(stroke_1[1, 0])  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    stroke_r.append(stroke_1[1, 1])  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    stroke_distance.append(sall[1])  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    stroke_position.append(stroke_3[1, :])
                    stroke_point.append(sall_points[1, :])
                elif sall[2] <= stroke_1[2, 1]:
                    stroke_height.append(stroke_1[2, 0])  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    stroke_r.append(stroke_1[2, 1])  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    stroke_distance.append(sall[2])  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    stroke_position.append(stroke_3[2, :])
                    stroke_point.append(sall_points[2, :])
                else:
                    stroke_height.append(0)  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    stroke_r.append(np.nan)  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    stroke_distance.append(np.nan)  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    stroke_position.append(np.array(["Indirect", "ground", np.nan, np.nan, np.nan]))
                    stroke_point.append([np.nan, np.nan])

                # 当pc的2个相高度一样时，判断两条pc的重叠区，离哪个pc近落在哪个pc
                if stroke_position[-1][1] == "phase conductor" and yc[0, 0] == yc[0, 1]:
                    # 点到pc的第1相的距离大于点到pc的第2相的距离
                    if sc > sb:
                        stroke_distance.append(sb)  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        stroke_position[-1][2] = circlep[i][0, 1]
                        stroke_position[-1][3] = span_npa[i][0, 1]
                        stroke_position[-1][4] = conductor[1, 0]
                        stroke_point[-1] = np.array([E, F])
                    else:
                        stroke_distance.append(sc)  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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
                rc = Phase1[0] * I ** Phase1[1]
                rg = Phase2[0] * I ** Phase2[1]
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
                sc = point_to_line_distance(C, D, P)  # 判断点到pc唯一1相的距离
                stroke_2 = [["Direct", "shield wire", circles[i][0], sw_npa[i][0], 1],
                            ["Direct", "phase conductor", circlep[i][0, 0], span_npa[i][0, 0], conductor[0, 0]]]
                # 谁高谁先判断-stroke_1 stroke_3 sall_points
                indices = np.argsort(-stroke_1[:, 0])
                stroke_1 = stroke_1[indices, :]
                stroke_3 = np.array(stroke_2)[indices]
                sall = np.array([ss, sc])  # 点到sw,pc唯一1相的距离
                sall = sall[indices]
                sall_points = np.array([[A, B], [C, D]])  # 跟哪条线段的sw或者pc有关系
                sall_points = sall_points[indices]
                if sall[0] <= stroke_1[0, 1]:
                    stroke_height.append(stroke_1[0, 0])  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    stroke_r.append(stroke_1[0, 1])  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    stroke_distance.append(sall[0])  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    stroke_position.append(stroke_3[0, :])
                    stroke_point.append(sall_points[0, :])
                elif sall[1] <= stroke_1[1, 1]:
                    stroke_height.append(stroke_1[1, 0])  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    stroke_r.append(stroke_1[1, 1])  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    stroke_distance.append(sall[1])  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    stroke_position.append(stroke_3[1, :])
                    stroke_point.append(sall_points[1, :])
                else:
                    stroke_height.append(0)  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    stroke_r.append(np.nan)  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    stroke_distance.append(np.nan)  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    stroke_position.append(np.array(["Indirect", "ground", np.nan, np.nan, np.nan]))
                    stroke_point.append([np.nan, np.nan])


        # sw有2根
        else:
            A, B = span_min[i][:2]  # shield wire的第一根的起点终点
            U, V = span_min[i][3:5]  # shield wire的第二根的起点终点
            ss = np.zeros((1, 2))  # 判断点到shield wire的第i根距离
            ss[0, 0] = point_to_line_distance(A, B, P)
            ss[0, 1] = point_to_line_distance(U, V, P)
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
                rc = Phase1[0] * I ** Phase1[1]
                rg = Phase2[0] * I ** Phase2[1]
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

                sc = point_to_line_distance(C, D, P)  # 判断点到pc的第1相的距离
                sb = point_to_line_distance(E, F, P)  # 判断点到pc的第2相的距离
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
                            ["Direct", "phase conductor", circlep[i][0, 1], span_npa[i][0, 1],
                             conductor[1, 0]]]  # pc的第1相,pc的第2相
                # 谁高谁先判断-stroke_1 stroke_3 sall_points
                indices = np.argsort(-stroke_1[:, 0])
                stroke_1 = stroke_1[indices, :]
                stroke_3 = np.array(stroke_2)[indices]
                sall = np.array([ss[0, 0], ss[0, 1], sc, sb])  # 点到sw的第1根,sw的第2根,pc的第1相,pc的第2相的距离
                sall = sall[indices]
                sall_points = np.array([[A, B], [U, V], [C, D], [E, F]])  # 跟哪条线段的sw或者pc有关系
                sall_points = sall_points[indices]
                if sall[0] <= stroke_1[0, 1]:
                    stroke_height.append(stroke_1[0, 0])  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    stroke_r.append(stroke_1[0, 1])  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    stroke_distance.append(sall[0])  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    stroke_position.append(stroke_3[0, :])
                    stroke_point.append(sall_points[0, :])
                elif sall[1] <= stroke_1[1, 1]:
                    stroke_height.append(stroke_1[1, 0])  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    stroke_r.append(stroke_1[1, 1])  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    stroke_distance.append(sall[1])  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    stroke_position.append(stroke_3[1, :])
                    stroke_point.append(sall_points[1, :])
                elif sall[2] <= stroke_1[2, 1]:
                    stroke_height.append(stroke_1[2, 0])  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    stroke_r.append(stroke_1[2, 1])  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    stroke_distance.append(sall[2])  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    stroke_position.append(stroke_3[2, :])
                    stroke_point.append(sall_points[2, :])
                elif sall[3] <= stroke_1[3, 1]:
                    stroke_height.append(stroke_1[3, 0])  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    stroke_r.append(stroke_1[3, 1])  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    stroke_distance.append(sall[3])  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    stroke_position.append(stroke_3[3, :])
                    stroke_point.append(sall_points[3, :])
                else:
                    stroke_height.append(0)  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    stroke_r.append(np.nan)  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    stroke_distance.append(np.nan)  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    stroke_position.append(np.array(["Indirect", "ground", np.nan, np.nan, np.nan]))
                    stroke_point.append([np.nan, np.nan])

                # 当sw的2根高度一样时，判断两条sw的重叠区，离哪个sw近落在哪个sw
                if stroke_position[-1][1] == "shield wire" and yg[0, 0] == yg[0, 1]:
                    # 点到sw的第1相的距离大于点到sw的第2相的距离
                    if ss[0, 0] > ss[0, 1]:
                        stroke_distance.append(ss[0,1])  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        stroke_position[-1][2] = circles[i][1]
                        stroke_position[-1][4] = conductor[1, 0]
                        stroke_point[-1] = np.array([U, V])
                    else:
                        stroke_distance.append(ss[0,0])  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        stroke_position[-1][2] = circles[i][0]
                        stroke_position[-1][4] = conductor[0, 0]
                        stroke_point[-1] = np.array([A, B])

                # 当pc的2个相高度一样时，判断两条pc的重叠区，离哪个pc近落在哪个pc
                if stroke_position[-1][1] == "phase conductor" and yc[0, 0] == yc[0, 1]:
                    # 点到pc的第1相的距离大于点到pc的第2相的距离
                    if sc > sb:
                        stroke_distance.append(sb)  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        stroke_position[-1][2] = circlep[i][0, 1]
                        stroke_position[-1][3] = span_npa[i][0, 1]
                        stroke_position[-1][4] = conductor[1, 0]
                        stroke_point[-1] = np.array([E, F])
                    else:
                        stroke_distance.append(sc)  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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
                rc = Phase1[0] * I ** Phase1[1]
                rg = Phase2[0] * I ** Phase2[1]
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

                sc = point_to_line_distance(C, D, P)  # 判断点到pc唯一1相的距离
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
                            ["Direct", "phase conductor", circlep[i][0, 0], span_npa[i][0, 0],
                             conductor[0, 0]]]  # pc唯一1相
                # 谁高谁先判断-stroke_1 stroke_3 sall_points
                indices = np.argsort(-stroke_1[:, 0])
                stroke_1 = stroke_1[indices, :]
                stroke_3 = np.array(stroke_2)[indices]
                sall = np.array([ss[0, 0], ss[0, 1], sc])  # 点到sw的第1根,sw的第2根,pc唯一1相的距离
                sall = sall[indices]
                sall_points = np.array([[A, B], [U, V], [C, D]])  # 跟哪条线段的sw或者pc有关系
                sall_points = sall_points[indices]
                if sall[0] <= stroke_1[0, 1]:
                    stroke_height.append(stroke_1[0, 0])  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    stroke_r.append(stroke_1[0, 1])  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    stroke_distance.append(sall[0])  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    stroke_position.append(stroke_3[0, :])
                    stroke_point.append(sall_points[0, :])
                elif sall[1] <= stroke_1[1, 1]:
                    stroke_height.append(stroke_1[1, 0])  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    stroke_r.append(stroke_1[1, 1])  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    stroke_distance.append(sall[1])  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    stroke_position.append(stroke_3[1, :])
                    stroke_point.append(sall_points[1, :])
                elif sall[2] <= stroke_1[2, 1]:
                    stroke_height.append(stroke_1[2, 0])  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    stroke_r.append(stroke_1[2, 1])  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    stroke_distance.append(sall[2])  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    stroke_position.append(stroke_3[2, :])
                    stroke_point.append(sall_points[2, :])
                else:
                    stroke_height.append(0)  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    stroke_r.append(np.nan)  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    stroke_distance.append(np.nan)  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    stroke_position.append(np.array(["Indirect", "ground", np.nan, np.nan, np.nan]))
                    stroke_point.append([np.nan, np.nan])

                # 当sw的2根高度一样时，判断两条sw的重叠区，离哪个sw近落在哪个sw
                if stroke_position[-1][1] == "shield wire" and yg[0, 0] == yg[0, 1]:
                    # 点到sw的第1相的距离大于点到sw的第2相的距离
                    if ss[0, 0] > ss[0, 1]:
                        stroke_distance.append(ss[0,1])  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        stroke_position[-1][2] = circles[i][1]
                        stroke_position[-1][4] = conductor[1, 0]
                        stroke_point[-1] = np.array([U, V])
                    else:
                        stroke_distance.append(ss[0,0])  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        stroke_position[-1][2] = circles[i][0]
                        stroke_position[-1][4] = conductor[0, 0]
                        stroke_point[-1] = np.array([A, B])

    # 判断斜坡是否存在!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if 'Slope_angle' in Line:
        for i in range(points_need.shape[0]):
            P = points_need[i, :]  # 待判断的落雷点
            # lightning current,单位是KA
            I = parameterst[i, 2]  # I=待判断的落雷点的电路参数的Ip
            # 斜坡的截面是无限长
            # 斜坡的起点Z-点P到线段ABZ的垂直交点Z
            AZ=slopestart[i][0]
            BZ=slopestart[i][1]
            APZ = P - AZ
            ABZ = BZ - AZ
            tZ = np.dot(APZ, ABZ) / np.dot(ABZ, ABZ)
            Z = AZ + tZ * ABZ
            #左右两个斜坡，哪个斜坡距离更近判断哪个斜坡
            A_angle = Line['Slope_angle'][dmin_all[i],0]  #左边的角度
            B_angle = Line['Slope_angle'][dmin_all[i],1]  #右边的角度
            # 计算角度A和B对应的斜率
            slope_A = np.tan(np.radians(A_angle))
            slope_B = np.tan(np.radians(B_angle))
            # 点到射线的距离公式
            def point_to_ray(x1, y1, slope, x2, y2):
                return np.abs(slope * (x1 - x2) - y1 + y2) / np.sqrt(slope**2 + 1)

            # 计算点P到斜线A的距离
            distance_A = point_to_ray(P[0],P[1], slope_A, Z[0], Z[1])
            # 计算点P到斜线B的距离
            distance_B = point_to_ray(P[0],P[1], slope_B, Z[0], Z[1])
            # 判断距离哪个斜坡更近
            if distance_A < distance_B:
                ca=np.deg2rad(A_angle) #转换为弧度
            else:
                ca=np.deg2rad(B_angle) #转换为弧度

            # 真正要判断的落雷点D是点到斜坡的距离=斜坡的吸引半径rs
            # 用公式计算吸引半径-斜坡
            rs = slopep[0] * I ** slopep[1]
            # 斜坡的斜率
            slope = np.tan(ca)
            # 求点C的坐标 (点A做垂线与斜坡交点)
            # 斜坡方程: y = slope * (x - B_x) + B_y
            C_x = P[0]
            C_y = slope * (C_x - Z[0]) + Z[1]
            C = np.array([C_x, C_y])
            # 求点D的坐标
            # 点D在斜坡的法线方向，法线斜率是 -1/slope
            # 点D与点E之间的距离为r，计算点D的坐标
            dx = rs / np.sqrt(1 + (1 / slope) ** 2)  # 沿法线方向的x偏移量
            dy = dx / slope  # 使用法线的正确斜率 (-1/slope)
            # 点D的坐标 (沿着斜坡的法线方向)
            D_x = C[0] + dx
            D_y = C[1] + dy
            D = np.array([D_x, D_y])
            # 求点E的坐标-垂足点的 XY 坐标
            # 联立法线方程和斜坡方程，计算E_x
            E_x = (D_y + (1 / slope) * D_x - Z[1] + slope * Z[0]) / (slope + 1 / slope)
            # 通过斜坡方程计算E_y
            E_y = slope * (E_x - Z[0]) + Z[1]
            E = np.array([E_x, E_y])
            # 原来的吸引位置是ground变成斜坡，落雷位置是点P
            if stroke_position[i][0] == "Indirect":
                stroke_position[i] =np.array(["Indirect", "scope", np.nan, np.nan, np.nan])
                stroke_point[i] = np.array([P[0], P[1]])
            else:  # 如果点D到原来的吸引位置的距离小于等于吸引半径，最终落雷点还是原来的吸引位置，不变
                line_segment23 = stroke_point[i]
                A23, B23 = line_segment23[0, :], line_segment23[1, :]
                sc = point_to_line_distance(A23, B23, D)  # 判断点D到原来的吸引位置的距离
                # 否则，最终落雷点是斜坡，落雷位置是点P（因为离地高度已经很近，所以落雷位置假设不变是最开始的位置）
                if sc>stroke_r[i]:
                    stroke_position[i] = np.array(["Indirect", "scope", np.nan, np.nan, np.nan])
                    stroke_point[i] = np.array([P[0], P[1]])

    # 判断building是否存在!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if 'buildings' in Line:
        for i in range(points_need.shape[0]):
            P = points_need[i, :]  # 待判断的落雷点
            # lightning current,单位是KA
            I = parameterst[i, 2]  # I=待判断的落雷点的电路参数的Ip
            stroke_1n = []
            stroke_2n = []
            stroke_3n = []
            stroke_5n = []
            salln = []
            # 遍历所有建筑物，使用 Line['buildings']
            for idx, building in enumerate(Line['buildings']):
                # 计算雷电位置到building的四条边的距离。
                # 计算建筑物的四个角点坐标
                bottom_left = building['building_XY'][0,:] #左下角
                bottom_right = building['building_XY'][1,:]  # 右下角
                top_left = building['building_XY'][2,:]  # 左上角
                top_right = building['building_XY'][3,:] # 右上角
                # 计算雷电位置到建筑物四条边的距离
                distance_to_bottom = point_to_line_distance(bottom_left, bottom_right, P)  # 底边
                distance_to_top = point_to_line_distance(top_left, top_right, P)  # 顶边
                distance_to_left = point_to_line_distance(bottom_left, top_left, P)  # 左边
                distance_to_right = point_to_line_distance(bottom_right, top_right, P)  # 右边
                # 雷电位置到建筑物的距离等于到建筑物四条边的最近的距离
                min_distance = min(distance_to_bottom, distance_to_top, distance_to_left, distance_to_right)
                # building的中心点的XYZ坐标
                center_x = np.mean([bottom_left[0], bottom_right[0], top_left[0], top_right[0]])
                center_y = np.mean([bottom_left[1], bottom_right[1], top_left[1], top_right[1]])
                center_point = np.array([center_x, center_y])
                # 用公式计算吸引半径
                rb = buildingp[0] * I ** buildingp[1]
                # 将每个building加入到比较列表中，标记建筑物编号
                # 矩阵的元素是同一数据类型，stroke_1n第一列是高度，第二列是吸引半径;stroke_2是雷击点的位置名称-span_npa
                stroke_1n.append([building['building_height'], rb])
                stroke_2n.append(["Indirect", f"building{idx+1}", np.nan, np.nan, np.nan])
                # building的落雷点是点P（因为离地高度已经很近，所以落雷位置假设不变是最开始的位置）
                stroke_5n.append([P[0], P[1]]) #列表形式
                salln.append(min_distance) # 点到building的距离

            # 将原来的吸引位置加入到比较列表
            stroke_1n.append([stroke_height[i], stroke_r[i]])
            stroke_2n.append(stroke_position[i])
            stroke_5n.append(stroke_point[i])
            salln.append(stroke_distance[i])  # 原来的吸引位置的距离
            # 谁高谁先判断-stroke_1 stroke_3 sall_points
            stroke_1n = np.array(stroke_1n)
            indices = np.argsort(-stroke_1n[:, 0])
            stroke_1n = stroke_1n[indices, :]
            stroke_3n = np.array(stroke_2n)[indices]
            stroke_5n = [stroke_5n[idx] for idx in indices]
            salln = np.array(salln)[indices]
            for j in range(len(stroke_1n)):
                if salln[j] <= stroke_1n[j, 1]:# 如果距离小于等于吸引半径
                    stroke_position[i]=stroke_3n[j, :]
                    stroke_point[i]=stroke_5n[j]
                    break # 满足距离小于等于吸引半径，即找到落雷点的最终位置后退出循环

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
    # stroke_point改为slope/building(间接雷)-坐标是点P（最开始的落雷位置points_need），sw/pc(直接雷)-点到线段的垂直交点;
    # segment_position2是雷的落雷点是Tower(1)/Span(2)/ground(0)
    # point_medium是落在有关线段（sw/pc）的3等分的哪个等分点
    # dmin_allID2是Tower/Span的ID
    stroke_point=[list(item) for item in stroke_point]
    points_need2 = [list(row) for row in points_need]
    stroke_position3 = np.stack(stroke_position)
    stroke_position21 = [stroke_position3[:, i] for i in range(stroke_position3.shape[1])]
    stroke_position2 = [list(arr) for arr in stroke_position21]
    dmin_allID2 = dmin_allID + 1  # python是0-based，实际情况的序号是从1开始
    segmentID2 = segmentID
    stroke_result = [list(a) for a in zip(points_need2, stroke_position2[0], stroke_position2[1],
                                          stroke_position2[2], stroke_position2[3], stroke_position2[4],
                                          stroke_point,
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
    sitesw = [index for index, value in enumerate(stroke_result[3]) if value == '1001']
    sitepc = [index for index, value in enumerate(stroke_result[3]) if value == '3001']
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
    # 检查文件是否存在并删除
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
                                                    'Tower_or_Span_ID', 'Circle_ID', 'phase_ID', 'conductor_ID',
                                                    'segment_ID',
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
                                               'Tower_or_Span_ID', 'Circle_ID', 'phase_ID', 'conductor_ID',
                                               'segment_ID',
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
                                                   'Tower_or_Span_ID', 'Circle_ID', 'phase_ID', 'conductor_ID',
                                                   'segment_ID',
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

    return resultstro, dataSTS, stroke_result