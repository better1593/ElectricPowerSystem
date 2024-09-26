import pandas as pd

from Model import Strategy
from Model.Network import Network
import matplotlib.pyplot as plt
import numpy as np
if __name__ == '__main__':
    # 1. 接收到创建新电网指令
    file_name = "01_8"
    varied_frequency = np.logspace(0, 37, 9)
    # strategy = Strategy.variant_frequency()
    strategy = Strategy.NonLinear()
    network = Network()
    network.run(file_name, strategy)
    print(network.solution)
    Y13_Tower_1 = [abs(i)for i in network.solution.loc["Y13_Tower_1"].tolist()]
    Y02_Tower_1 = [abs(i) for i in network.solution.loc["Y02_Tower_1"].tolist()]

    X05 = network.solution.loc["X05"].tolist()
    X08 = network.solution.loc["X08"].tolist()
    min = [abs(a) for a in abs(X05 - X08)]

    lightning_source = network.sources.loc["X05"].tolist()[:network.Nt]
    # 生成数据
    x = np.arange(0, network.Nt, 1)  # 横坐标数据为从0到10之间，步长为0.1的等差数组
    y = Y13_Tower_1  #

    # 生成图形
    plt.plot(x, lightning_source)
    plt.title('lightning_source')
    plt.show()
    plt.plot(x, y)
    plt.title('ins_i')
    plt.show()
    plt.plot(x, Y02_Tower_1)
    plt.title('top_i')
    plt.show()
    plt.plot(x, min)
    plt.title('ins_v')
    plt.show()

    # 显示图形
    # 2. 接收到所需测试的类型
    #strategy1 = Strategy.baseStrategy()
   # network.update_H()

# y13_tower1
#x05,x08
    from scipy.io import loadmat
    import pandas as pd
    file_path = "C:\\Users\\demo\\Desktop\\PolyU\\电路参数生成矩阵\\VTS\\03Caculate for Large System\\Tower_V9c_yeung\\Tower_V9c_ding2\\集中参数接地\\"
    bran_index_t1 = ['Y0' + str(i+1)+'_Tower_1' for i in range(8)]
    bran_index_t1.extend(['Y13_Tower_1', 'Y14_Tower_1', 'Y15_Tower_1', 'Y09_Tower_1'])
    bran_index_t2 = ['Y0' + str(i+1)+'_Tower_2' for i in range(8)]
    bran_index_t2.extend(['Y13_Tower_2', 'Y14_Tower_2', 'Y15_Tower_2', 'Y09_Tower_2'])
    bran_index_t1.extend(bran_index_t2)
    # bran_index_t1.extend(['Y1001S_splited_1', 'Y3001A_splited_1', 'Y3001B_splited_1', 'Y3001C_splited_1'])
    bran_index_t1.extend(['Y1001S', 'Y3001A', 'Y3001B', 'Y3001C'])
    bran_index_t1.extend(['Y_T12O_S', 'Y_T22O_S', 'Y_T12O_A', 'Y_T22O_A', 'Y_T12O_B', 'Y_T22O_B', 'Y_T12O_C', 'Y_T22O_C'])
    bran_index = bran_index_t1






