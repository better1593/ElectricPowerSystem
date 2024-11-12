import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import os
import pandas as pd
from collections import Counter

def lighting_parameters_distribution_S (allp,MC_lgtn,DSave,Line,foldname):
    # 绘图
    plt.close('all')

    # struct获取每个变量名
    mode = MC_lgtn['mode']
    muN = MC_lgtn['muN']
    sigmaN = MC_lgtn['sigmaN']
    rhoi = MC_lgtn['rhoi']
    fixn = MC_lgtn['fixn']
    stroke = MC_lgtn['stroke']
    muI = MC_lgtn['muI']
    mu = MC_lgtn['mu']
    sigma1st = MC_lgtn['sigma1st']
    sigma = MC_lgtn['sigma']
    userInput1 = DSave['userInput1']
    userInput2 = DSave['userInput2']
    userInput0 = DSave['userInput0']
    Coordinates = Line['Node']
    Edges = Line['Edges'].astype(int)
    swheigh = MC_lgtn['swheight']
    middlePoints = MC_lgtn['middlePoints']
    pn = MC_lgtn['pn']
    averageh = MC_lgtn['averageh']

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
            if stroke==1:
                sN=np.array([1.])

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
            if stroke==1:
                sN_init=np.array([1.])

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
    XY_need3 = np.array([
        [middlePoints[0], (allp+1) * averageh],
        [middlePoints[1], (allp+1) * averageh],
        [middlePoints[1], (allp - 1+1) * averageh],
        [middlePoints[0], (allp - 1+1) * averageh]
    ])
    XY_need3 = np.vstack([XY_need3, XY_need3[0, :]])

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

    xmin = np.min(XY_need3[:, 0])  # 左下角x坐标
    xmax = np.max(XY_need3[:, 0])  # 右上角x坐标
    ymin = np.min(XY_need3[:, 1])  # 左下角y坐标
    ymax = np.max(XY_need3[:, 1])  # 右上角y坐标
    # 画第2张图-杆塔图+包围线
    plt.figure(2)
    plt.plot(XY_need3[:, 0], XY_need3[:, 1], 'k-', linewidth=2)
    plt.plot(XY_need3[:, 0], XY_need3[:, 1], 'r*')
    plt.axis('equal')

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

    if userInput1 == 1:
        # 将double 数组转换为table
        df = pd.DataFrame(XY_need3, columns=['X_coordinate', 'Y_coordinate'])
        # 将文件夹名添加到文件路径
        outputFile = os.path.join(foldname, 'LGT Zone Boundary_S.xlsx')
        sheetName = f'Sheet{allp+1}'
        if not os.path.exists(outputFile):
            mode = 'w'
        else:
            mode = 'a'

        with pd.ExcelWriter(outputFile, engine='openpyxl', mode=mode) as writer:
            df.to_excel(writer, sheet_name=sheetName, index=False)

    if userInput2 == 1:
        # 将double 数组转换为table
        df2 = pd.DataFrame(parameterst, columns=['flash', 'stroke', 'Ip', 'tf', 'Sm', 'th'])
        # 将文件夹名添加到文件路径
        outputFile = os.path.join(foldname, 'Current Parameter_S.xlsx')
        sheetName = f'Sheet{allp + 1}'
        if not os.path.exists(outputFile):
            mode = 'w'
        else:
            mode = 'a'

        with pd.ExcelWriter(outputFile, engine='openpyxl', mode=mode) as writer:
            df2.to_excel(writer, sheet_name=sheetName, index=False)

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
        outputFile = os.path.join(foldname, 'Stroke Location_S.xlsx')
        sheetName = f'Sheet{allp + 1}'
        if not os.path.exists(outputFile):
            mode = 'w'
        else:
            mode = 'a'

        with pd.ExcelWriter(outputFile, engine='openpyxl', mode=mode) as writer:
            df0.to_excel(writer, sheet_name=sheetName, index=False)

    # 所有统计参数写成struct结构，用一个变量传递数据
    resultcur = {'parameterst': parameterst,
                 'points_need': points_need,
                 'siteone': siteone,
                 'unique_flash': unique_flash,  # 有哪些次flash
                 'stroke_counts': stroke_counts,
                 'area': polygonArea,
                 'flash_stroke': flash_stroke}  # 每次flash有多少个stroke

    return resultcur, parameterst, flash_stroke, points_need,XY_need3