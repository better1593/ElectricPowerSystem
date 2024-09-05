from Model import Strategy
from Model.Network import Network
import matplotlib.pyplot as plt
import numpy as np
if __name__ == '__main__':
    # 1. 接收到创建新电网指令
    file_name = "01_8"
    strategy = Strategy.NonLiear()
    network = Network()
    network.run(file_name, strategy)
    print(network.solution)

    Y13_Tower_1 = [abs(i)for i in network.solution.loc["Y13_Tower_1"].tolist()]
    X05 = network.solution.loc["X05"].tolist()
    X08 = network.solution.loc["X08"].tolist()
    min = [abs(a - b) for a, b in zip(X05, X08)]

    Y01_Tower_1 = [abs(i)for i in network.solution.loc["Y01_Tower_1"].tolist()]
    X05 = network.solution.loc["X01"].tolist()
  #  X08 = network.solution.loc["X08"].tolist()
  #  min = [abs(a - b) for a, b in zip(X05, X08)]


    t = 150
    x = np.arange(0, t, 1)
    fig, ax = plt.subplots()  # 创建图实例
    y1 = Y13_Tower_1[:t]
    ax.plot(x, y1, label='Current of Insulator')  # 作y1 = x 图，并标记此线名为linear
    y2 = Y01_Tower_1[:t]
    ax.plot(x, y2, label='Current of Tower')  # 作y2 = x^2 图，并标记此线名为quadratic
    y3 = network.sources.loc['X05'].tolist()[:t]
    ax.plot(x, y3, label='Waveform')  # 作y3 = x^3 图，并标记此线名为cubic

    ax.set_xlabel('Time (1e-6s)')  # 设置x轴名称 x label
    ax.set_ylabel('Current(A)')  # 设置y轴名称 y label
    ax.set_title('Current of Insulator/Tower/Stoke')  # 设置图名为Simple Plot
    ax.legend()  # 自动检测要在图例中显示的元素，并且显示

    plt.show()  # 图形可视化

    fig2, ax2 = plt.subplots()  # 创建图实例
    y4 = min[:t]
    ax2.plot(x, y4, label='Voltage of Insulator')  # 作y1 = x 图，并标记此线名为linear
    y5 = X05[:t]
    ax2.plot(x, y5, label='Voltage of Tower')  # 作y2 = x^2 图，并标记此线名为quadratic


    ax2.set_xlabel('Time (1e-6s)')  # 设置x轴名称 x label
    ax2.set_ylabel('Voltage(V)')  # 设置y轴名称 y label
    ax2.set_title('Voltage of Insulator/Tower')  # 设置图名为Simple Plot
    ax2.legend()  # 自动检测要在图例中显示的元素，并且显示

    plt.show()  # 图形可视化

