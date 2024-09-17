import pandas as pd

from Model import Strategy
from Model.Network import Network
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # 1. 接收到创建新电网指令
    file_name = "01_tube"
    varied_frequency = np.arange(0, 37, 9)
    network = Network()
    change = Strategy.Change_DE_max()
    network.run(file_name,change)


    print(network.solution)
    print(network.measurement)





