import pandas as pd
import pickle

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
    change = Strategy.Change_DE_max()
    network.run(load_dict,change)
    pd.DataFrame(network.run_measure()).to_csv("ROD.csv")



    network.sensitive_analysis(load_dict)


    print('end')
    #network.run_MC(load_dict)

   # pickle.dump(network, open("/Data/output/network.pkl", 'wb'))  # 序列化

   # print(network.solution)
    #print(network.measurement)





