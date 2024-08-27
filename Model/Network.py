import numpy as np
import pandas as pd

from Driver.initialization.initialization import initialize_OHL, initialize_tower, initial_source, initial_lump
from Driver.modeling.OHL_modeling import OHL_building
from Driver.modeling.tower_modeling import tower_building

from Model.Cable import Cable
from Model.Lightning import Lightning
from Model.Tower import Tower
from Model.Wires import OHLWire


class Network:
    def __init__(self, **kwargs):
        self.towers = kwargs.get('towers', [])
        self.cables = kwargs.get('cable', [])
        self.ohls = kwargs.get('ohls', [])
        self.sources = kwargs.get('sources', [])
        self.branches = {}
        self.starts = []
        self.ends = []
        self.H = pd.DataFrame()
        self.incidence_matrix = pd.DataFrame()
        self.resistance_matrix = pd.DataFrame()
        self.inductance_matrix = pd.DataFrame()
        self.potential_matrix = pd.DataFrame()
        self.capacitance_matrix = pd.DataFrame()
        self.conductance_martix = pd.DataFrame()

    def calculate_branches(self):
        wires = [tower.wires.get_all_wires() for tower in self.towers]
        wires2 = [ohl.wires.get_all_wires() for ohl in self.ohls]
        wires3 = [cable.wires.get_all_wires() for cable in self.cables]
        wires = wires + wires2 + wires3

        for wire in wires:
            startnode = [wire.start_node.x, wire.start_node.y, wire.start_node.z]
            endnode = [wire.end_node.x, wire.end_node.y, wire.end_node.z]
            self.branches[wire.name] = [startnode, endnode]
            self.starts.append(startnode)
            self.ends.append(endnode)


    def initalize_network(self):
        file_name = "01_2"
        max_length = 50
        self.towers = initialize_tower(file_name, max_length=max_length) #初始化tower
        self.ohls = initialize_OHL(file_name, max_length) #初始化ohl
        self.calculate_branches()

    def initialize_source(self):
        file_name = "01_2"
        nodes = self.capacitance_matrix.columns.tolist()
        self.sources = initial_source(self, nodes, file_name)


    def calculate_H(self,f0,frq_default,max_length):
        self.initalize_network()
        segment_num = int(3)  # 正常情况下，segment_num由segment_length和线长反算，但matlab中线长参数位于Tower中，在python中如何修改？
        segment_length = 20  # 预设的参数
        tower_building(self.towers[0], f0, max_length)
        OHL_building(self.ohls[0], frq_default, segment_num, segment_length)

        for tower in self.towers:
            self.concate_matrix(tower)
        for ohl in self.ohls:
            self.concate_matrix(self.ohls[0])
        for cable in self.cables:
            self.concate_matrix(cable)


    def concate_matrix(self,obj):

        self.incidence_matrix = self.incidence_matrix.add(obj.incidence_matrix, fill_value=0).fillna(0)
        self.resistance_matrix = self.resistance_matrix.add(obj.resistance_matrix, fill_value=0).fillna(0)
        self.inductance_matrix = self.inductance_matrix.add(obj.inductance_matrix, fill_value=0).fillna(0)
        self.capacitance_matrix = self.capacitance_matrix.add(obj.capacitance_matrix, fill_value=0).fillna(0)
        self.conductance_martix = self.conductance_martix.add(obj.conductance_martix, fill_value=0).fillna(0)

        print("得到一个合并的大矩阵H（a，b）")
    def initial_souces(self):

        print("得到源u")

    def get_x(self):
        print("x=au+bu结果？")

    def update_H(self,h):
        print("更新H矩阵")

    def combine_parameter_martix(self):
        #按照towers，cables，ohls顺序合并参数矩阵
        #合并sources矩阵
        incidence_matrix = pd.DataFrame()
        resistance_matrix = pd.DataFrame()
        inductance_matrix = pd.DataFrame()
        capacitance_matrix = pd.DataFrame()
        conductance_martix = pd.DataFrame()
        voltage_source_martix = pd.DataFrame()
        current_source_martix = pd.DataFrame()
        for model_list in [self.towers, self.cables, self.ohls]:
            for model in model_list:
                incidence_matrix.add(model.incidence_matrix, fill_value=0).fillna(0)
                resistance_matrix.add(model.resistance_matrix, fill_value=0).fillna(0)
                inductance_matrix.add(model.inductance_matrix, fill_value=0).fillna(0)
                capacitance_matrix.add(model.capacitance_matrix, fill_value=0).fillna(0)
                conductance_martix.add(model.conductance_martix, fill_value=0).fillna(0)
                voltage_source_martix.add(model.voltage_source_martix, fill_value=0).fillna(0)
                current_source_martix.add(model.current_source_martix, fill_value=0).fillna(0)

        return incidence_matrix, resistance_matrix, inductance_matrix, capacitance_matrix, conductance_martix, voltage_source_martix, current_source_martix


    #
    # def solution(self.ima, imb, R, L, G, C, sources, dt, Nt):
    #     """
    #     【函数功能】电路求解
    #     【入参】
    #     ima(numpy.ndarray:Nbran*Nnode)：关联矩阵A（Nbran：支路数，Nnode：节点数）
    #     imb(numpy.ndarray:Nbran*Nnode)：关联矩阵B（Nbran：支路数，Nnode：节点数）
    #     R(numpy.ndarray:Nbran*Nbran)：电阻矩阵（Nbran：支路数）
    #     L(numpy.ndarray:Nbran*Nbran)：电感矩阵（Nbran：支路数）
    #     G(numpy.ndarray:Nnode*Nnode)：电导矩阵（Nnode：节点数）
    #     C(numpy.ndarray:Nnode*Nnode)：电容矩阵（Nnode：节点数）
    #     sources(numpy.ndarray:(Nbran+Nnode)*Nt)：电源矩阵（Nbran：支路数，Nnode：节点数）
    #     dt(float)：步长
    #     Nt(int)：计算总次数
    #
    #     【出参】
    #     out(numpy.ndarray:(Nbran+Nnode)*Nt)：计算结果矩阵（Nbran：支路数，Nnode：节点数）
    #     """
    #     Nbran, Nnode = ima.shape
    #     out = np.zeros((Nbran+Nnode, Nt))
    #     for i in range(Nt - 1):
    #         Vnode = out[:Nnode, i]
    #         Ibran = out[Nnode:, i]
    #         LEFT = np.block([[-ima, -R - L / dt], [G + C / dt, -imb]])
    #         inv_LEFT = np.linalg.inv(LEFT)
    #         RIGHT = np.block([[(-L / dt).dot(Ibran)], [(C / dt).dot(Vnode)]])
    #         temp_result = inv_LEFT.dot(sources + RIGHT)
    #         out[:, i + 1] = np.copy(temp_result)
    #     return out
