import matplotlib.pyplot as plt
import numpy as np

def Plotting(GLB, Vnt_total, Ibt_total):
    t = np.arange(1, GLB['Nt'] + 1)

    # 第一张图
    plt.figure()
    plt.plot(t, Vnt_total[34, :])
    plt.show()

    # 第二张图
    plt.figure()
    plt.plot(t, Vnt_total[32, :], t, Vnt_total[52, :], t, Vnt_total[72, :], t, Vnt_total[92, :], t, Vnt_total[112, :], '-.')
    plt.show()

    # 第三张图
    plt.figure()
    plt.plot(t, Ibt_total[50, :], t, Ibt_total[70, :], t, Ibt_total[90, :], t, Ibt_total[110, :])
    plt.show()

    # 第四张图
    plt.figure()
    plt.plot(t, Ibt_total[30, :], t, Ibt_total[50, :], t, Ibt_total[70, :], t, Ibt_total[90, :], t, Ibt_total[110, :])
    plt.show()

    # 第五张图
    plt.figure()
    plt.plot(t, Vnt_total[30, :], t, Vnt_total[90, :], t, Vnt_total[150, :], t, Vnt_total[200, :], t, Vnt_total[210, :])
    plt.show()

    # 第六张图
    plt.figure()
    plt.plot(t, Vnt_total[31, :], t, Vnt_total[51, :], t, Vnt_total[71, :], t, Vnt_total[91, :], t, Vnt_total[111, :])
    plt.show()

    # 第七张图
    plt.figure()
    plt.plot(t, Vnt_total[30, :], t, Vnt_total[31, :], t, Vnt_total[32, :], t, Vnt_total[33, :])
    plt.show()