import pandas as pd

from Model import Strategy
from Model.Network import Network
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # 1. 接收到创建新电网指令
    file_name = "01_8"
    varied_frequency = np.arange(0, 37, 9)
    strategy = Strategy.baseStrategy()
    network = Network()
    network.run(file_name, strategy)
   # measure_result = network.run_measurement()
    print(network.solution)





