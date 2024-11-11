import pandas as pd
import pickle
import multiprocessing
from multiprocessing import Process, Manager
from Model import Strategy
from Model.Network import Network
import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
if __name__ == '__main__':
    # 1. 接收到创建新电网指令
    file_name = "01_8"
    json_file_path = "Data/input/" + file_name + ".json"
    # 0. read json file
    with open(json_file_path, 'r', encoding="utf-8") as j:
        load_dict = json.load(j)
    varied_frequency = np.arange(0, 37, 9)
    network = Network()
    #change = Strategy.Change_DE_max()
    strategy = Strategy.variant_frequency()
    network.run(load_dict,strategy)
    #network.run_individual(load_dict)
    #pd.DataFrame(network.run_measure()).to_csv("Data/Output/"+file_name+"_output.csv")


    # 二、灵敏度分析模块
    #network.sensitive_analysis(load_dict)

    # 三、蒙特卡洛，大量模拟运算

    #network.run_MC(load_dict)

#    pickle.dump(network, open("network.pkl", 'wb'))  # 序列化

    #print(network.solution)
    #print(network.measurement)





