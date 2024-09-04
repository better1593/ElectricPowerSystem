from Model import Strategy
from Model.Network import Network
if __name__ == '__main__':
    # 1. 接收到创建新电网指令
    file_name = "01_7"
    strategy = Strategy.NonLiear()
    network = Network()
    network.run(file_name, strategy)
    print(network.solution)

    # 2. 接收到所需测试的类型
    #strategy1 = Strategy.baseStrategy()
   # network.update_H()
