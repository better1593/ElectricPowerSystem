import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import numpy as np
import pandas as pd
from functools import reduce
from itertools import chain
import copy
from Driver.initialization.initialization import initialize_OHL, initialize_tower, initial_lightning, initial_lump, \
    initialize_cable, initialize_ground
from Driver.modeling.OHL_modeling import OHL_building, OHL_building_variant_frequency
from Driver.modeling.cable_modeling import cable_building, cable_building_variant_frequency
from Driver.modeling.tower_modeling import tower_building, tower_building_variant_frequency
from Function.Calculators.InducedVoltage_calculate import InducedVoltage_calculate, LightningCurrent_calculate, \
    H_MagneticField_calculate, ElectricField_calculate, ElectricField_above_lossy, InducedVoltage_calculate_indirect,InducedVoltage_calculate_direct
from Risk_Evaluate.MC import run_MC
from Model.Cable import Cable
from Model.Lightning import Lightning
from Model.Tower import Tower
from Model.Wires import OHLWire
from Utils.Math import distance, segment_branch
import Model.Strategy as Strategy
from Model.Contant import Constant
from Model.Lightning import Stroke,Lightning,Channel
import math
from multiprocessing import Process, Manager,Lock


class Network:
    def __init__(self, **kwargs):
        self.towers = kwargs.get('towers', [])
        self.cables = kwargs.get('cables', [])
        self.OHLs = kwargs.get('OHLs', [])
        self.sources = pd.DataFrame()
        self.branches = {}
        self.starts = []
        self.ends = []
        self.H = {}
        self.solution = pd.DataFrame()
        self.measurement = {}
        self.incidence_matrix_A = pd.DataFrame()
        self.incidence_matrix_B = pd.DataFrame()
        self.resistance_matrix = pd.DataFrame()
        self.inductance_matrix = pd.DataFrame()
        self.capacitance_matrix = pd.DataFrame()
        self.conductance_matrix = pd.DataFrame()
        self.voltage_source_matrix = pd.DataFrame()
        self.current_source_matrix = pd.DataFrame()
        self.solution_type = {
            'linear': False,
            'constant_step': True,
            'variable_frequency': False
        }
        self.Nfit=9
        self.f0 = 2e4
        self.max_length = 200
        # self.varied_frequency = np.logspace(0, 9, 37)
        self.varied_frequency = np.array([])
        for i in range(6):
            temp = np.linspace(5e-2*10**i, 5e-1*10**i, 10)
            self.varied_frequency = np.hstack((self.varied_frequency, temp))
        self.global_ground = 0
        self.ground = None
        self.dt =None
        self.T = None
        self.VF = None
        self.frq = None
        self.switch_disruptive_effect_models = []
        self.voltage_controled_switchs = []
        self.time_controled_switchs = []
        self.nolinear_resistors = []
        self.lightning = None
        self.VF_dict = {}
    #记录电网元素之间的关系
    def tower_branches(self,branches):
        tower_nodes = []
        for tower in self.towers:
            for wire in list(tower.wires.get_all_wires().values()):
                startnode = {wire.start_node.name: [wire.start_node.x, wire.start_node.y, wire.start_node.z]}
                endnode = {wire.end_node.name: [wire.end_node.x, wire.end_node.y, wire.end_node.z]}
                tower_nodes.append(wire.start_node.name)
                tower_nodes.append(wire.end_node.name)
                branches[wire.name] = [startnode, endnode, tower.name]
        return branches, set(tower_nodes)
    def OHL_branches(self,branches,maxlength):
        OHL_nodes = []
        for obj in self.OHLs:
            wires = list(obj.wires.get_all_wires().values())
            for wire in wires:
                position_obj_start = {wire.start_node.name: [wire.start_node.x ,
                                                             wire.start_node.y ,
                                                             wire.start_node.z]}
                position_obj_end = {wire.end_node.name: [wire.end_node.x ,
                                                         wire.end_node.y ,
                                                         wire.end_node.z ]}
                Nt = int(np.ceil(distance(obj.info.HeadTower_pos, obj.info.TailTower_pos) / maxlength))
                OHL_nodes.append(wire.start_node.name)
                OHL_nodes.append(wire.end_node.name)
                branches[wire.name] = [position_obj_start, position_obj_end, obj.info.name, Nt]
        return branches, set(OHL_nodes)
    def cable_branches(self,branches,maxlength):
        cable_nodes = []
        for obj in self.cables:
            wires = list(obj.wires.get_all_wires().values())
            for wire in wires:
                position_obj_start = {wire.start_node.name: [wire.start_node.x ,
                                                             wire.start_node.y ,
                                                             wire.start_node.z]}
                position_obj_end = {wire.end_node.name: [wire.end_node.x ,
                                                         wire.end_node.y ,
                                                         wire.end_node.z ]}
                Nt = int(np.ceil(distance(obj.info.HeadTower_pos, obj.info.TailTower_pos) / maxlength))
                cable_nodes.append(wire.start_node.name)
                cable_nodes.append(wire.end_node.name)
                branches[wire.name] = [position_obj_start, position_obj_end, obj.info.name, Nt]
        return branches, set(cable_nodes)
    def calculate_branches(self, maxlength):
        branches = {}
        branches,tb = self.tower_branches(branches)
        branches,ob = self.OHL_branches(branches,maxlength)
        branches,cb = self.cable_branches(branches,maxlength)
        return branches

    def tower_initial(self,load_dict):
        if 'Tower' in load_dict:
            self.towers = [initialize_tower(tower, max_length=self.max_length,dt=self.dt,T = self.T,VF=self.VF) for tower in load_dict['Tower']]
            self.measurement = reduce(lambda acc,tower:{**acc,**tower.Measurement},self.towers,{})
        for tower in self.towers:
            vf = 0
            gnd = self.ground if self.global_ground == 1 else tower.ground
            if tower.info.Mode_Con ==1 or tower.info.Mode_Gnd ==2:
                vf = 1
            if vf==1:
                print("tower apply verified frequent")
                tower_building_variant_frequency(tower, self.f0, gnd, self.varied_frequency, self.Nfit, self.dt)
            else:
                tower_building(tower, self.f0, gnd)
            self.switch_disruptive_effect_models.extend(tower.lump.switch_disruptive_effect_models)
            self.voltage_controled_switchs.extend(tower.lump.voltage_controled_switchs)
            self.time_controled_switchs.extend(tower.lump.time_controled_switchs)
            self.nolinear_resistors.extend(tower.lump.nolinear_resistors)
            for device_list in [tower.devices.insulators, tower.devices.arrestors, tower.devices.transformers]:
                for device in device_list:
                    self.switch_disruptive_effect_models.extend(device.switch_disruptive_effect_models)
                    self.voltage_controled_switchs.extend(device.voltage_controled_switchs)
                    self.time_controled_switchs.extend(device.time_controled_switchs)
                    self.nolinear_resistors.extend(device.nolinear_resistors)
    # initialize internal network elements
    def OHL_initial(self,load_dict):
        if 'OHL' in load_dict:
            self.OHLs = [initialize_OHL(ohl, max_length=self.max_length) for ohl in load_dict['OHL']]
    def OHL_building(self):
        for ohl in self.OHLs:
            vf = 0
            gnd = self.ground if self.global_ground == 1 else ohl.ground
            if ohl.info.model1 ==1 or ohl.info.model2 ==2:
                vf = 1
            if vf==1:
                print("OHL apply verified frequent")
                OHL_building_variant_frequency(ohl, self.max_length, gnd, self.varied_frequency, self.Nfit, self.dt)
            else:
                OHL_building(ohl, self.max_length, gnd, self.f0)
    def cable_initial(self,load_dict,VF):
        if 'Cable' in load_dict:
            self.cables = [initialize_cable(cable, max_length=self.max_length,VF=VF) for cable in load_dict['Cable']]
    def cable_building(self):
        for cable in self.cables:
            vf = 0
            gnd = self.ground if self.global_ground == 1 else cable.ground
            if cable.info.Mode_Con ==1 or cable.info.Mode_Gnd ==2:
                vf = 1
            if vf==1:
                print("Cable apply verified frequent")
                cable_building_variant_frequency(cable, gnd, self.varied_frequency, self.dt) #1,2
            else:
                cable_building(cable, gnd, self.f0)
    def initialize_network(self, load_dict, varied_frequency,VF,dt, T):

        self.tower_initial(load_dict)
        self.OHL_initial(load_dict)
        self.OHL_building()
        self.cable_initial(load_dict,VF)
        self.cable_building()


        # 2. build dedicated matrix for all elements
        # segment_num = int(3)  # 正常情况下，segment_num由segment_length和线长反算，但matlab中线长参数位于Tower中，在python中如何修改？
        # segment_length = 50  # 预设的参数
    def source_initial(self,load_dict,nodes,branches,constants,share_dict):

        if load_dict["Source"]["Lightning"]:
            light = load_dict["Source"]["Lightning"]
            self.lightning = initial_lightning(light, dt=self.dt)
            if light["area"].split("_")[0] == "OHL":
                for ohl in load_dict["OHL"]:
                    if ohl["name"]==light["area"]:
                        wires = ohl["Wire"]
                        for wire in wires:
                            cir_id = wire['cir_id']
                            phase_id = wire['phase_id']
                            if cir_id == light["cir_id"] and phase_id == light["phase_id"]:
                                if wire['type'] == 'SW':
                                    bran = 'Y' + str(cir_id) + 'S'
                                    U_out,I_out = self.source_calculate(self.lightning, light["area"], bran, light["position"],nodes,branches,constants)
                                    sources = self.add_lump(U_out,I_out)
                                    return sources
                                elif wire['type'] == 'CIRO':
                                    bran = 'Y' + str(cir_id) + wire['phase']
                                    U_out,I_out = self.source_calculate(self.lightning, light["area"], bran, light["position"],nodes,branches,constants)
                                    sources = self.add_lump(U_out, I_out)
                                    return sources
            if light["area"].split("_")[0] == "tower":
                U_out,I_out,share_dict = self.source_calculate(self.lightning,
                                                 light["area"], light["wire"], light["position"],nodes,branches,constants,share_dict)
                sources = self.add_lump(U_out, I_out)
                return sources

    #输出U/I
    def source_calculate(self,lightning,area,wire,position,nodes,branches,constants,shared_dict):
        branches = segment_branch(branches)
        start = [list(l[0].values())[0] for l in list(branches.values())]
        end = [list(l[1].values())[0] for l in list(branches.values())]
        branches = list(branches.keys())  # 让branches只存储支路的列表，可节省内存
        pt_start = np.array(start)
        pt_end = np.array(end)

        U_out = pd.DataFrame()
        I_out = pd.DataFrame()
        if lightning.type =="Indirect":
            for i in range(len(lightning.strokes)):

                Er_lossy = 0
                Ez_lossy = 0
                erg = constants.epr
                sigma_g = constants.sigma
                if (erg, sigma_g,0) in shared_dict:
                    Er_lossy = shared_dict[(erg, sigma_g,0)]
                    Ez_lossy = shared_dict[(erg, sigma_g,1)]
                    print("------------existing ---------------")
                else:
                    H_p = H_MagneticField_calculate(pt_start, pt_end, lightning.strokes[i],
                                                    lightning.channel,
                                                    constants.ep0, constants.vc)  # 计算磁场
                    Ez_T, Er_T = ElectricField_calculate(pt_start, pt_end, lightning.strokes[i],
                                                         lightning.channel,
                                                         constants.ep0, constants.vc)  # 计算电场
                    # 计算有损地面的电场
                    Er_lossy = ElectricField_above_lossy(-H_p, Er_T, constants, shared_dict, constants.sigma)
                    shared_dict[(erg, sigma_g,0)] = Er_lossy
                    Ez_lossy = Ez_T
                    shared_dict[(erg, sigma_g,1)] = Ez_lossy
                new_U = InducedVoltage_calculate_indirect(pt_start, pt_end, branches, lightning,
                                                    stroke_sequence=i,Er_lossy=Er_lossy,Ez_lossy=Ez_lossy)
                U_out = pd.concat([U_out,new_U],axis=1,ignore_index=True)
                I_out = pd.concat(
                    [I_out, LightningCurrent_calculate(area,wire,position,self,nodes, lightning, stroke_sequence=i)],
                    axis=1,ignore_index=True)
        if lightning.type =="Direct":
            for i in range(len(lightning.strokes)):
                new_U = InducedVoltage_calculate_direct(branches,lightning,i)
                U_out = pd.concat([U_out,new_U],axis=1,ignore_index=True)
                I_out = pd.concat(
                    [I_out, LightningCurrent_calculate(area,wire,position,self,nodes, lightning, stroke_sequence=i)],
                    axis=1,ignore_index=True)
        # Source_Matrix = pd.concat([I_out, U_out], axis=0)
        return U_out,I_out,shared_dict
    #U/I矩阵 加上Lump的U/I，输出source
    def add_lump(self,U_out,I_out):
        lumps = [tower.lump for tower in self.towers]
        devices = [tower.devices for tower in self.towers]
        for lump in lumps:
            U_out = U_out.add(lump.voltage_source_matrix, fill_value=0).fillna(0)
            I_out = I_out.add(lump.current_source_matrix, fill_value=0).fillna(0)
        for lumps in list(map(lambda device: device.arrestors + device.insulators + device.transformers, devices)):
            for lump in lumps:
                U_out = U_out.add(lump.voltage_source_matrix, fill_value=0).fillna(0)
                I_out = I_out.add(lump.current_source_matrix, fill_value=0).fillna(0)
        return pd.concat([U_out, I_out], axis=0)
    #R,L,G,C矩阵合并
    def tower_matrix(self):
        for tower in self.towers:
            self.incidence_matrix_A = self.incidence_matrix_A.add(tower.incidence_matrix_A, fill_value=0).fillna(0)
            self.incidence_matrix_B = self.incidence_matrix_B.add(tower.incidence_matrix_B, fill_value=0).fillna(0)
            self.resistance_matrix = self.resistance_matrix.add(tower.resistance_matrix, fill_value=0).fillna(0)
            self.inductance_matrix = self.inductance_matrix.add(tower.inductance_matrix, fill_value=0).fillna(0)
            self.capacitance_matrix = self.capacitance_matrix.add(tower.capacitance_matrix, fill_value=0).fillna(0)
            self.conductance_matrix = self.conductance_matrix.add(tower.conductance_matrix, fill_value=0).fillna(0)
        self.build_H()
    def OHL_matrix(self):
        for ohl in self.OHLs:
            self.incidence_matrix_A = self.incidence_matrix_A.add(ohl.incidence_matrix, fill_value=0).fillna(0)
            self.incidence_matrix_B = self.incidence_matrix_B.add(ohl.incidence_matrix, fill_value=0).fillna(0)
            self.resistance_matrix = self.resistance_matrix.add(ohl.resistance_matrix, fill_value=0).fillna(0)
            self.inductance_matrix = self.inductance_matrix.add(ohl.inductance_matrix, fill_value=0).fillna(0)
            self.capacitance_matrix = self.capacitance_matrix.add(ohl.capacitance_matrix, fill_value=0).fillna(0)
            self.conductance_matrix = self.conductance_matrix.add(ohl.conductance_matrix, fill_value=0).fillna(0)
    def cable_matrix(self):
        for cable in self.cables:
            self.incidence_matrix_A = self.incidence_matrix_A.add(cable.incidence_matrix, fill_value=0).fillna(0)
            self.incidence_matrix_B = self.incidence_matrix_B.add(cable.incidence_matrix, fill_value=0).fillna(0)
            self.resistance_matrix = self.resistance_matrix.add(cable.resistance_matrix, fill_value=0).fillna(0)
            self.inductance_matrix = self.inductance_matrix.add(cable.inductance_matrix, fill_value=0).fillna(0)
            self.capacitance_matrix = self.capacitance_matrix.add(cable.capacitance_matrix, fill_value=0).fillna(0)
            self.conductance_matrix = self.conductance_matrix.add(cable.conductance_matrix, fill_value=0).fillna(0)
    def combine_parameter_matrix(self):

        # 按照towers，cables，ohls顺序合并参数矩阵
        self.tower_matrix()
        self.OHL_matrix()
        self.cable_matrix()

        self.build_H()
    def build_H(self):
        self.H["incidence_matrix_A"] = copy.deepcopy(self.incidence_matrix_A)
        self.H["incidence_matrix_B"] = copy.deepcopy(self.incidence_matrix_B)
        self.H["resistance_matrix"] = copy.deepcopy(self.resistance_matrix)
        self.H["inductance_matrix"] = copy.deepcopy(self.inductance_matrix)
        self.H["capacitance_matrix"] = copy.deepcopy(self.capacitance_matrix)
        self.H["conductance_matrix"] = copy.deepcopy(self.conductance_matrix)
        self.H["switch_disruptive_effect_models"] = copy.deepcopy(self.switch_disruptive_effect_models)
        self.H["voltage_controled_switchs"] = copy.deepcopy(self.voltage_controled_switchs)
        self.H["time_controled_switchs"] = copy.deepcopy(self.time_controled_switchs)
        self.H["nolinear_resistors"] = copy.deepcopy(self.nolinear_resistors)
        return self.H
    def reset_matrix(self):
       self.incidence_matrix_A = self.H["incidence_matrix_A"]
       self.incidence_matrix_B  = self.H["incidence_matrix_B"]
       self.resistance_matrix = self.H["resistance_matrix"]
       self.inductance_matrix = self.H["inductance_matrix"]
       self.capacitance_matrix =  self.H["capacitance_matrix"]
       self.conductance_matrix = self.H["conductance_matrix"]
    #执行不同的算法：线性/非线性
    def calculate(self,T,dt,H,sources):
        if not self.switch_disruptive_effect_models and not self.voltage_controled_switchs and not self.time_controled_switchs and not self.nolinear_resistors:
            strategy = Strategy.Linear()
        else:
            strategy = Strategy.NonLinear()
        return strategy.apply(T,dt,H,sources)
    #设置全局参数
    def global_set(self,load_dict):
        self.frq = np.concatenate([
            np.arange(1, 91, 10),
            np.arange(100, 1000, 100),
            np.arange(1000, 10000, 1000),
            np.arange(10000, 100000, 10000)
        ])
        self.VF = {'odc': 10,
                   'frq': self.frq}
        # self.dt = 1e-8
        # self.T = 0.003
        # 是否有定义
        if 'Global' in load_dict:
            self.dt = load_dict['Global']['delta_time']
            self.T = load_dict['Global']['time']
            f0 = load_dict['Global']['constant_frequency']
            self.f0 = np.array([f0]).reshape(-1)
            self.max_length = load_dict['Global']['max_length']
            self.global_ground = load_dict['Global']['ground']['glb']
            self.ground = initialize_ground(load_dict['Global']['ground']) if 'ground' in load_dict['Global'] else None
            self.Nt = int(np.ceil(self.T / self.dt))
    def OHL_calculate(self,load_dict, ohl_nodes, ohl_branches,solution_nodes):
        # TODO:

        self.OHL_building()

        sources = self.source_initial(load_dict, ohl_nodes, ohl_branches)
        ohl_solution = self.calculate(self.Nt,self.dt, self.H, sources)
        ohl_solution_nodes = ohl_solution.loc[[list(solution_nodes)]]

    # 基础模块 分布运行
    def run_individual(self,load_dict):
        self.global_set(load_dict)
        constants = Constant()
        constants.ep0 = 8.85e-12
        # 1. 计算Tower矩阵
        self.tower_initial(load_dict)#tower出初始化和矩阵构建
        self.tower_matrix() #合并tower矩阵
        self.OHL_initial(load_dict)

        tower_branches = {}
        ohl_branches = {}
        # 2.
        tower_branches, tower_nodes = self.tower_branches(tower_branches)
        ohl_branches, ohl_nodes = self.OHL_branches(ohl_branches,self.max_length)

        tower_ohl_nodes = tower_nodes.intersection(ohl_nodes)

        # 3. tower - source calculate
        sources = self.source_initial(load_dict, list(tower_nodes),tower_branches,constants)

        tower_solution = self.calculate(self.Nt,self.dt,self.H,sources)

        tower_solution_nodes = tower_solution.loc[[list(tower_ohl_nodes)]]
    # 基础模块合并运行
    def run(self,load_dict,*basestrategy):
        # 0. 手动预设值
        self.global_set(load_dict)
        # self.dt = 1e-6
        # self.T = 1e-5
        # self.Nt = int(np.ceil(self.T / self.dt))
        constants = Constant()
        constants.ep0 = 8.85e-12
        # 1. 初始化电网，根据电网信息计算源
        self.initialize_network(load_dict, self.frq,self.VF,self.dt,self.T)
        self.combine_parameter_matrix()

        # 2. 保存支路节点信息(for source calculate)
        # 合并计算的时候是这样设置
        branches = self.calculate_branches(self.max_length)
        nodes = self.capacitance_matrix.columns.tolist()

        # 3. 初始化源，计算结果
        share_dict = {}
        start_time = time.time()  # 记录开始时间
        sources = self.source_initial(load_dict, nodes,branches,constants,share_dict)
        end = time.time()  # 记录开始时间
        print(f"Total running time: {end-start_time} seconds")  # 打印运行时长
        solution = self.calculate(self.Nt,self.dt,self.H,sources)
        end2 = time.time()  # 记录开始时间
        print(f"Total running time: {end2 - end} seconds")  # 打印运行时长
        print(solution)
    def sensitive_analysis(self,load_dict):


        if load_dict["Sensitivity_analysis"]["Stroke"]["position"]:
            wire = load_dict["Sensitivity_analysis"]["Stroke"]["area"]
            area = load_dict["Sensitivity_analysis"]["Stroke"]["area"]
            if area.split("_")[0]=="tower":
                for obj in load_dict["Tower"]:
                    for w in obj["Wire"]:
                        if load_dict["Sensitivity_analysis"]["Stroke"]["wire"]:
                            if w["name"] ==wire:
                                Strategy.Change_light_pos().apply(self,self.lightning,
                                                  load_dict["Sensitivity_analysis"]["Stroke"]["area"],
                                                  w,
                                                load_dict["Sensitivity_analysis"]["Stroke"]["position"])
            if area.split("_")[0] == "OHL":
                for obj in load_dict["OHL"]:
                    for w in obj["Wire"]:
                        if load_dict["Sensitivity_analysis"]["Stroke"]["cir_id"]:
                            if w["cir_id"] ==load_dict["Sensitivity_analysis"]["Stroke"]["cir_id"]\
                                    and w["phase"] ==load_dict["Sensitivity_analysis"]["Stroke"]["phase"]:
                                Strategy.Change_light_pos().apply(self, self.lightning,
                                                                  load_dict["Sensitivity_analysis"]["Stroke"]["area"],
                                                                  w,
                                                                  load_dict["Sensitivity_analysis"]["Stroke"]["position"])
                                break
                    break
        if load_dict["Sensitivity_analysis"]["Stroke"]["waveform"]:
            Strategy.Change_light_waveform().apply(self,load_dict["Source"]["Lightning"],
                                                   load_dict["Sensitivity_analysis"]["Stroke"]["waveform"],
                                                   load_dict)

        if load_dict["Sensitivity_analysis"]["Stroke"]["paramenters"]:
            Strategy.Change_light_waveform().apply(self, load_dict["Source"]["Lightning"],
                                                   load_dict["Sensitivity_analysis"]["Stroke"]["paramenters"],
                                                   load_dict)

        if load_dict["Sensitivity_analysis"]["Arrester"]["name"]:
            name = load_dict["Sensitivity_analysis"]["Arrester"]["name"]
            Strategy.Change_Arrestor().apply(self, load_dict,name)

        if load_dict["Sensitivity_analysis"]["SW"]["name"]:
            name = load_dict["Sensitivity_analysis"]["SW"]["name"]
            Strategy.Change_SW().apply(self,load_dict,name)

        if load_dict["Sensitivity_analysis"]["DE"]:
            Strategy.Change_DE_max().apply(self,load_dict["Sensitivity_analysis"]["DE"])
            self.calculate(self.Nt, self.H, self.sources)
            pd.DataFrame(self.run_measure()).to_csv("DE_modified.csv")


        if load_dict["Sensitivity_analysis"]["ROD"]:
            for tower in load_dict["Tower"]:
                if tower["name"] ==load_dict["Sensitivity_analysis"]["ROD"]["tower"]:
                    Strategy.Change_ROD().apply(self,load_dict,
                                        load_dict["Sensitivity_analysis"]["ROD"]["lump"],
                                        load_dict["Sensitivity_analysis"]["ROD"]["r"],
                                        load_dict["Sensitivity_analysis"]["ROD"]["l"])

        print("you are starting sensitive analysis")
    def run_measure(self):
        # lumpname/branname: label(bran0/lump1),probe,branname,n1,n2,(towername)
        if self.measurement:
            return Strategy.Measurement().apply(measurement=self.measurement, solution=self.solution,dt=self.dt)
    def run_MC(self,load_dict):
        # 0. 手动预设值
        self.global_set(load_dict)
        self.dt = 1e-8
        #self.Nt = 1000
        self.T = 2e-5
        self.Nt = int(np.ceil(self.T / self.dt))
        # 1. 初始化电网，根据电网信息计算源
        self.initialize_network(load_dict, self.frq,self.VF,self.dt,self.T)
        self.combine_parameter_matrix()

        # 2. 保存支路节点信息
        branches = self.calculate_branches(self.max_length)
        nodes = self.capacitance_matrix.columns.tolist()
        # 3. 生成多个雷电
        if load_dict["MC"]:
            print("running Monte Carlo to generate lightnings")
            df27,parameterst,stroke_result = run_MC(self,load_dict)
            index = 0
            MC_result = []
            for i in df27.groupby("flash"):
                stroke_list = []
                for j in range(i[1].shape[0]):
                    stroke_type = "Heidler"
                    duration = self.T
                    dt = self.dt
                    stroke = Stroke(stroke_type, duration=duration, dt=dt, is_calculated=True, parameter_set=None,
                                    parameters=parameterst[index].tolist()[2:])
                    stroke.calculate()
                    index += 1
                    stroke_list.append(stroke)
                flash_type = stroke_result[0][index - 1]
                area = int(stroke_result[1][index - 1])
                area_id = 0 if math.isnan(stroke_result[2][index - 1]) else str(int(stroke_result[2][index - 1]))
                cir_id = 0 if math.isnan(float(stroke_result[3][index - 1])) else int(
                    float(stroke_result[3][index - 1]))
                phase_id = 0 if math.isnan(float(stroke_result[4][index - 1])) else int(
                    float(stroke_result[4][index - 1]))
                position_xy = stroke_result[8][index - 1]
                position = None
                wire = None
                if area == 0:
                    area = "Ground"
                    position = position_xy.append(0)
                elif area == 1:
                    area = "tower_" + area_id
                    for tower in load_dict["Tower"]:
                        if tower["Info"]["name"] == area:
                            z = tower["Info"]["pole_height"]
                            position = position_xy.append(z)
                            for w in tower["Wire"]:
                                if w["pos_1"][2] == z or w["pos_2"][2] == z:
                                    wire = w["bran"]
                            if wire is None:
                                wire = tower["Wire"][0]
                elif area == 2:
                    area = "OHL_" + area_id
                    for ohl in load_dict["OHL"]:
                        if ohl["Info"]["name"] == area:
                            for w in ohl["Wire"]:
                                cir_id_ohl = w['cir_id']
                                phase_id_ohl = w['phase_id']
                                if cir_id_ohl == cir_id and phase_id_ohl == phase_id:
                                    z = w["node1_pos"][2]
                                    position = position_xy.append(z)
                                    if w['type'] == 'SW':
                                        wire = 'Y' + str(cir_id) + 'S'
                                    elif w['type'] == 'CIRO':
                                        wire = 'Y' + str(cir_id) + w['phase']

                lightning = Lightning(id=1, type=flash_type, strokes=stroke_list, channel=Channel(position_xy))
                # for stroke in lightning.strokes:
                #     stroke.duration = self.T
                #     stroke.Nt = self.Nt
                #     stroke.t_us = np.array(list(range(self.Nt))) * self.dt
                MC_result.append((lightning, area, wire, position_xy))

            with Manager() as manager:
                shared_dict = manager.dict()  # 创建一个共享字典
                self.run_multiprocessed(MC_result, nodes, branches,shared_dict)
    def run_multiprocessed(self, MC_results, nodes, branches,shared_dict):
        # 创建一个Manager对象，用于创建共享字典
        # with Manager() as manager:
        #     shared_dict = manager.dict()  # 创建一个共享字典
        #lock = Lock()
        # 创建进程列表
        start_time = time.time()  # 记录开始时间
        #processes = []

        for index,MC in enumerate(MC_results):
            # 创建Process对象，传递当前实例和MC_result的元素
            # p = Process(target=process_item, args=(MC, nodes, branches, self,index,shared_dict))
            # processes.append(p)
            # p.start()  # 启动进程
            process_item(MC, nodes, branches, self,index,shared_dict)


        # 等待所有进程完成
        # for p in processes:
        #     p.join()
        end_time = time.time()  # 记录结束时间
        duration = end_time - start_time  # 计算运行时长
        print(f"Total running time: {duration} seconds")  # 打印运行时长

        # 保存运行时长到文件
        with open("runtime_limit_time.txt", "w") as f:
            f.write(f"Total running time: {duration} seconds\n")

        return duration

def process_item(MC, nodes, branches, self_ref,index,shared_dict):
    constants = Constant()
    constants.ep0 = 8.85e-12
    # start_time = time.time()  # 记录开始时间
    U_out, I_out,shared_dict = self_ref.source_calculate(MC[0], MC[1], MC[2], MC[3], nodes, branches,constants,shared_dict)
    sources = self_ref.add_lump(U_out, I_out)
    # end_time = time.time()  # 记录结束时间
    # duration = end_time - start_time  # 计算运行时长
    # print(f"Source running time: {duration} seconds")  # 打印运行时长
    H = self_ref.build_H()
    solution = self_ref.calculate(self_ref.Nt,self_ref.dt, H, sources)
    #print(solution)
    print("calculate"+str(index))
    # end_time2 = time.time()  # 记录结束时间
    # duration2 = end_time2 - end_time  # 计算运行时长
    # print(f"Calculation running time: {duration2} seconds")  # 打印运行时长

