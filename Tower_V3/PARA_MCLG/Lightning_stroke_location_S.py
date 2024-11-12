import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import re

from Tower_V3.PARA_MCLG.OTHS.point_to_line_distance import point_to_line_distance

def Lightning_stroke_location_S (allp,Line,resultcur,DSave,foldname):
    # struct获取每个变量名
    Coordinates = Line['Node']
    Edges = Line['Edges'].astype(int)
    points_need = resultcur['points_need']
    parameterst = resultcur['parameterst']
    siteone = resultcur['siteone']
    unique_flash = resultcur['unique_flash']
    stroke_counts = resultcur['stroke_counts']
    polygonArea = resultcur['area']
    OHLP = Line['Suppose_OHLP2']
    segments = Line['segments']
    SWnumber = Line['SWnumber']
    userInput3 = DSave['userInput3']
    userInput3d = DSave['userInput3d']
    userInput3ind = DSave['userInput3ind']
    flash_stroke = resultcur['flash_stroke']

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
        span0 = OHLP[dmin_index]
        span_min.append([span0[1, 0:2], span0[1, 3:5], span0[1, 2]])

    dmin_all=dmin_all.astype(int)
    # EGM model
    # 判断每个点points_need雷击点的位置-stroke_result
    stroke_position = []
    stroke_point = []
    for i in range(points_need.shape[0]):
        P = points_need[i, :]  # 待判断的落雷点
        # lightning current,单位是KA
        I = parameterst[i, 2]  # I=待判断的落雷点的电路参数的Ip
        rg = 5.5 * I ** 0.65
        A, B = span_min[i][:2]
        ss = point_to_line_distance(A, B, P)
        yg = span_min[i][2]
        if ss <= rg:
            stroke_position.append(np.array(["Direct","shield wire",1001,1,1]))
            stroke_point.append(np.array([A, B]))
        else:
            stroke_position.append(np.array(["Indirect", "ground", np.nan, np.nan, np.nan]))
            stroke_point.append([np.nan, np.nan])

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
    stroke_point = [list(item) for item in stroke_point]
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
    # plt.show()

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
    outputFile = os.path.join(foldname, 'Statistics Summary_S.xlsx')
    sheetName = f'Sheet{allp + 1}'
    if not os.path.exists(outputFile):
        mode = 'w'
    else:
        mode = 'a'

    with pd.ExcelWriter(outputFile, engine='openpyxl', mode=mode) as writer:
        STS.to_excel(writer, sheet_name=sheetName, index=False)

    # save
    if userInput3 == 1:
        # 将double 数组转换为table
        df3 = pd.DataFrame(stroke_result5, columns=['Direct_or_Indirect', 'Tower1_or_Span2_or_Ground0',
                                                    'Tower_or_Span_ID', 'Circle_ID', 'phase_ID', 'conductor_ID',
                                                    'segment_ID',
                                                    'coordinates_of_point_XY',
                                                    'Location_of_the_lightning_strike_point_XY'])
        # 将文件夹名添加到文件路径
        outputFile = os.path.join(foldname, 'Stroke Position_S.xlsx')
        sheetName = f'Sheet{allp + 1}'
        if not os.path.exists(outputFile):
            mode = 'w'
        else:
            mode = 'a'

        with pd.ExcelWriter(outputFile, engine='openpyxl', mode=mode) as writer:
            df3.to_excel(writer, sheet_name=sheetName, index=False)

    if userInput3d == 1:
        # 将double 数组转换为table
        df3d = pd.DataFrame(stroke_d, columns=['Direct_or_Indirect', 'Tower1_or_Span2_or_Ground0',
                                               'Tower_or_Span_ID', 'Circle_ID', 'phase_ID', 'conductor_ID',
                                               'segment_ID',
                                               'coordinates_of_point_XY',
                                               'Location_of_the_lightning_strike_point_XY'])
        # 将文件夹名添加到文件路径
        outputFile = os.path.join(foldname, 'DIR Stroke Position_S.xlsx')
        sheetName = f'Sheet{allp + 1}'
        if not os.path.exists(outputFile):
            mode = 'w'
        else:
            mode = 'a'

        with pd.ExcelWriter(outputFile, engine='openpyxl', mode=mode) as writer:
            df3d.to_excel(writer, sheet_name=sheetName, index=False)

    if userInput3ind == 1:
        # 将double 数组转换为table
        df3ind = pd.DataFrame(stroke_ind, columns=['Direct_or_Indirect', 'Tower1_or_Span2_or_Ground0',
                                                   'Tower_or_Span_ID', 'Circle_ID', 'phase_ID', 'conductor_ID',
                                                   'segment_ID',
                                                   'coordinates_of_point_XY',
                                                   'Location_of_the_lightning_strike_point_XY'])
        # 将文件夹名添加到文件路径
        outputFile = os.path.join(foldname, 'IND Stroke Position_S.xlsx')
        sheetName = f'Sheet{allp + 1}'
        if not os.path.exists(outputFile):
            mode = 'w'
        else:
            mode = 'a'

        with pd.ExcelWriter(outputFile, engine='openpyxl', mode=mode) as writer:
            df3ind.to_excel(writer, sheet_name=sheetName, index=False)

    # 关于输出添加一列[flash id, stroke number, 直接（1）或间接（0）]
    DIND = stroke_result[0]
    DINDs = np.array([1 if x == 'Direct' else 0 for x in DIND])
    DIND2 = DINDs[resultcur['siteone']]
    flash_stroke = np.hstack((flash_stroke, DIND2.reshape(-1, 1)))
    # 将double 数组转换为table
    df27 = pd.DataFrame(flash_stroke, columns=['flash', 'stroke', 'Direct1_Indirect2'])
    # 将文件夹名添加到文件路径
    outputFile = os.path.join(foldname, 'Flash_Stroke Dist_S.xlsx')
    sheetName = f'Sheet{allp + 1}'
    if not os.path.exists(outputFile):
        mode = 'w'
    else:
        mode = 'a'

    with pd.ExcelWriter(outputFile, engine='openpyxl', mode=mode) as writer:
        df27.to_excel(writer, sheet_name=sheetName, index=False)

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