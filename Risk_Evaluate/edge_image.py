import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from matplotlib.path import Path
from scipy.spatial import distance
import os
import pandas as pd

def edge_image(Line, DSave, LatDis_max, foldname):
    # struct获取每个变量名
    Coordinates = Line['Node']
    Edges = Line['Edges'].astype(int)
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
