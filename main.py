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



