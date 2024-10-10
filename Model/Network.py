import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from functools import reduce
from itertools import chain
import copy
from Driver.initialization.initialization import initialize_OHL, initialize_tower, initial_source, initial_lump, \
    initialize_cable, initialize_ground
from Driver.modeling.OHL_modeling import OHL_building
from Driver.modeling.cable_modeling import cable_building
from Driver.modeling.tower_modeling import tower_building
from Function.Calculators.InducedVoltage_calculate import InducedVoltage_calculate, LightningCurrent_calculate
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

        self.f0 = 2e4
        self.max_length = 20
        self.varied_frequency = np.arange(0, 37, 9)
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
    #记录电网元素之间的关系
    def calculate_branches(self, maxlength):
        tower_branch_node = {}
        tower_nodes = []
        for tower in self.towers:
            for wire in list(tower.wires.get_all_wires().values()):
                startnode = {wire.start_node.name: [wire.start_node.x, wire.start_node.y, wire.start_node.z]}
                endnode = {wire.end_node.name: [wire.end_node.x, wire.end_node.y, wire.end_node.z]}
                tower_nodes.append(startnode)
                tower_nodes.append(endnode)
                self.branches[wire.name] = [startnode, endnode, tower.name]

        for obj in self.OHLs + self.cables:
            wires = list(obj.wires.get_all_wires().values())
            for wire in wires:
                position_obj_start = {wire.start_node.name: [wire.start_node.x ,
                                                             wire.start_node.y ,
                                                             wire.start_node.z]}
                # position_tower_start = self.towers.get(obj.info.HeadTower).info.position
                # start_position = [x + y for x, y in zip(position_obj_start, position_tower_start)]
                position_obj_end = {wire.end_node.name: [wire.end_node.x ,
                                                         wire.end_node.y ,
                                                         wire.end_node.z ]}
                # position_tower_end = self.towers.get(obj.info.TailTower).info.position
                # end_position = [x + y for x, y in zip(position_obj_end, position_tower_end)]
                Nt = int(np.ceil(distance(obj.info.HeadTower_pos, obj.info.TailTower_pos) / maxlength))
                self.branches[wire.name] = [position_obj_start, position_obj_end, obj.info.name, Nt]

    def initial_tower(self,load_dict):
        if 'Tower' in load_dict:
            self.towers = [initialize_tower(tower, max_length=self.max_length,dt=self.dt,T = self.T,VF=self.VF) for tower in load_dict['Tower']]
            self.measurement = reduce(lambda acc,tower:{**acc,**tower.Measurement},self.towers,{})
        for tower in self.towers:
            gnd = self.ground if self.global_ground == 1 else tower.ground
            tower_building(tower, self.f0, self.max_length, gnd, self.frq)
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
    def initialize_network(self, load_dict, varied_frequency,VF,dt, T):

        self.initial_tower(load_dict)

        # 1. initialize all elements in the network

        #self.towers = reduce(lambda a, b: dict(a, **b), tower_list)
        if 'OHL' in load_dict:
            self.OHLs = [initialize_OHL(ohl, max_length=self.max_length) for ohl in load_dict['OHL']]
        if 'Cable' in load_dict:
            self.cables = [initialize_cable(cable, max_length=self.max_length,VF=VF) for cable in load_dict['Cable']]

        # 2. build dedicated matrix for all elements
        # segment_num = int(3)  # 正常情况下，segment_num由segment_length和线长反算，但matlab中线长参数位于Tower中，在python中如何修改？
        # segment_length = 50  # 预设的参数


        for ohl in self.OHLs:
            gnd = self.ground if self.global_ground == 1 else ohl.ground
            OHL_building(ohl, self.max_length, self.varied_frequency, gnd)
        for cable in self.cables:
            gnd = self.ground if self.global_ground == 1 else cable.ground
            cable_building(cable, self.f0, self.varied_frequency, gnd)

        # 3. combine matrix
        self.combine_parameter_matrix()

    # initialize external source
    def initialize_source(self, load_dict,dt):
        self.sources =initial_source( load_dict, dt=dt)


    def source_calculate(self,lightning,area,wire,position):
        nodes = self.capacitance_matrix.columns.tolist()

        branches = segment_branch(self.branches)
        start = [list(l[0].values())[0] for l in list(branches.values())]
        end = [list(l[1].values())[0] for l in list(branches.values())]
        branches = list(branches.keys())  # 让branches只存储支路的列表，可节省内存
        pt_start = np.array(start)
        pt_end = np.array(end)
        constants = Constant()
        constants.ep0 = 8.85e-12
        U_out = pd.DataFrame()
        I_out = pd.DataFrame()
        for i in range(len(lightning.strokes)):
            U_out = pd.concat([U_out, InducedVoltage_calculate(pt_start, pt_end, branches, lightning,
                                                               stroke_sequence=i, constants=constants)], axis=1,
                              ignore_index=True)
            I_out = pd.concat(
                [I_out, LightningCurrent_calculate(area,wire,position,self,nodes, lightning, stroke_sequence=i)],
                axis=1,ignore_index=True)
        # Source_Matrix = pd.concat([I_out, U_out], axis=0)
        lumps = [tower.lump for tower in self.towers]
        devices = [tower.devices for tower in self.towers]
        for lump in lumps:
            U_out = U_out.add(lump.voltage_source_matrix, fill_value=0).fillna(0)
            I_out = I_out.add(lump.current_source_matrix, fill_value=0).fillna(0)
        for lumps in list(map(lambda device: device.arrestors + device.insulators + device.transformers, devices)):
            for lump in lumps:
                U_out = U_out.add(lump.voltage_source_matrix, fill_value=0).fillna(0)
                I_out = I_out.add(lump.current_source_matrix, fill_value=0).fillna(0)

        return pd.concat([U_out,I_out],axis=0)

    #R,L,G,C矩阵合并
    def combine_parameter_matrix(self):

        # 按照towers，cables，ohls顺序合并参数矩阵
        for tower in self.towers:
            self.incidence_matrix_A = self.incidence_matrix_A.add(tower.incidence_matrix_A, fill_value=0).fillna(0)
            self.incidence_matrix_B = self.incidence_matrix_B.add(tower.incidence_matrix_B, fill_value=0).fillna(0)
            self.resistance_matrix = self.resistance_matrix.add(tower.resistance_matrix, fill_value=0).fillna(0)
            self.inductance_matrix = self.inductance_matrix.add(tower.inductance_matrix, fill_value=0).fillna(0)
            self.capacitance_matrix = self.capacitance_matrix.add(tower.capacitance_matrix, fill_value=0).fillna(0)
            self.conductance_matrix = self.conductance_matrix.add(tower.conductance_matrix, fill_value=0).fillna(0)

        for model_list in [self.OHLs, self.cables]:
            for model in model_list:
                self.incidence_matrix_A = self.incidence_matrix_A.add(model.incidence_matrix, fill_value=0).fillna(0)
                self.incidence_matrix_B = self.incidence_matrix_B.add(model.incidence_matrix, fill_value=0).fillna(0)
                self.resistance_matrix = self.resistance_matrix.add(model.resistance_matrix, fill_value=0).fillna(0)
                self.inductance_matrix = self.inductance_matrix.add(model.inductance_matrix, fill_value=0).fillna(0)
                self.capacitance_matrix = self.capacitance_matrix.add(model.capacitance_matrix, fill_value=0).fillna(0)
                self.conductance_matrix = self.conductance_matrix.add(model.conductance_matrix, fill_value=0).fillna(0)
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

    #执行不同的算法
    def calculate(self,dt,H,sources):
        if not self.switch_disruptive_effect_models and not self.voltage_controled_switchs and not self.time_controled_switchs and not self.nolinear_resistors:
            strategy = Strategy.Linear()
        else:
            strategy = Strategy.NonLinear()
        return strategy.apply(dt,H,sources)

    def run_measure(self):
        # lumpname/branname: label(bran0/lump1),probe,branname,n1,n2,(towername)
        if self.measurement:
            return Strategy.Measurement().apply(measurement=self.measurement, solution=self.solution,dt=self.dt)

    def run_MC(self,load_dict):
        # 0. 手动预设值
        self.frq = np.concatenate([
            np.arange(1, 91, 10),
            np.arange(100, 1000, 100),
            np.arange(1000, 10000, 1000),
            np.arange(10000, 100000, 10000)
        ])
        self.VF = {'odc': 10,
              'frq': self.frq}
        self.dt = 1e-8
        self.T = 0.003
        # 是否有定义
        if 'Global' in load_dict:
            self.dt = load_dict['Global']['delta_time']
            self.T = load_dict['Global']['time']
            f0 = load_dict['Global']['constant_frequency']
            max_length = load_dict['Global']['max_length']
            global_ground = load_dict['Global']['ground']['glb']
            ground = initialize_ground(load_dict['Global']['ground']) if 'ground' in load_dict['Global'] else None
        # 1. 初始化电网，根据电网信息计算源
        self.initialize_network(load_dict, self.frq,self.VF,self.dt,self.T)
        self.Nt = int(np.ceil(self.T/self.dt))


        # 2. 保存支路节点信息
        self.calculate_branches(self.max_length)

        # 3. 生成多个雷电
        if load_dict["MC"]:
            print("running Monte Carlo to generate lightnings")
            df27,parameterst,stroke_result = run_MC(self,load_dict)

            index = 0
            with ProcessPoolExecutor() as executor:
                futures = []
                # 为每个 flash 提交任务
                for i in df27.groupby("flash"):
                    futures.append(
                        executor.submit(self.process_flash, i, df27, parameterst, stroke_result, load_dict, index))

                # 等待所有任务完成
                for future in as_completed(futures):
                    print(future.result())


    def process_flash(self,i, df27, parameterst, stroke_result, load_dict, index):
        stroke_list = []
        for j in range(i[1].shape[0]):
            stroke_type = "Heidler"
            duration = 1e-5
            dt = 1e-8
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
        sources = self.source_calculate(lightning, area, wire, position_xy)
        H = copy.deepcopy(self.build_H())
        self.calculate(self.dt, H, sources)
        print("light: "+i+" calculating")


    def run(self,load_dict,*basestrategy):
        # 0. 手动预设值
        self.frq = np.concatenate([
            np.arange(1, 91, 10),
            np.arange(100, 1000, 100),
            np.arange(1000, 10000, 1000),
            np.arange(10000, 100000, 10000)
        ])
        self.VF = {'odc': 10,
              'frq': self.frq}
        self.dt = 1e-8
        self.T = 0.003
        # 是否有定义
        if 'Global' in load_dict:
            self.dt = load_dict['Global']['delta_time']
            self.T = load_dict['Global']['time']
            f0 = load_dict['Global']['constant_frequency']
            max_length = load_dict['Global']['max_length']
            global_ground = load_dict['Global']['ground']['glb']
            ground = initialize_ground(load_dict['Global']['ground']) if 'ground' in load_dict['Global'] else None
        # 1. 初始化电网，根据电网信息计算源
        self.initialize_network(load_dict, self.frq,self.VF,self.dt,self.T)
        self.Nt = int(np.ceil(self.T/self.dt))


        # 2. 保存支路节点信息
        self.calculate_branches(self.max_length)

        # 3. 初始化源，计算结果
        if load_dict["Source"]["Lightning"]:
            light = load_dict["Source"]["Lightning"]
            self.lightning = initial_source(light, dt=self.dt)
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
                                    self.sources = self.source_calculate(self.lightning, light["area"], bran, light["position"])
                                elif wire['type'] == 'CIRO':
                                    bran = 'Y' + str(cir_id) + wire['phase']
                                    self.sources = self.source_calculate(self.lightning, light["area"], bran, light["position"])
            if light["area"].split("_")[0] == "tower":
                self.sources = self.source_calculate(self.lightning,
                                                 light["area"], light["wire"], light["position"])

        self.solution = self.calculate(self.dt,self.H,self.sources)

    def find_wire(self,load_dict,light,area,cir,phase):
        if area.split("_")[0] == "OHL":
            for ohl in load_dict["OHL"]:
                if ohl["name"]==light["area"]:
                    wires = ohl["Wire"]
                    for wire in wires:
                        cir_id = wire['cir_id']
                        phase_id = wire['phase_id']
                        if cir_id == light["cir_id"] and phase_id == light["phase_id"]:
                            if wire['type'] == 'SW':
                                bran = 'Y' + str(cir_id) + 'S'
                                self.sources = self.source_calculate(self.lightning, light["area"], bran, light["position"])
                            elif wire['type'] == 'CIRO':
                                bran = 'Y' + str(cir_id) + wire['phase']
                                self.sources = self.source_calculate(self.lightning, light["area"], bran, light["position"])
        if light["area"].split("_")[0] == "tower":
            self.sources = self.source_calculate(self.lightning,
                                             light["area"], light["wire"], light["position"])

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
            node_map = load_dict["Sensitivity_analysis"]["Arrester"]["node_map"]
            wire_map = load_dict["Sensitivity_analysis"]["Arrester"]["wire_map"]
            Strategy.Change_Arrestor_pos().apply(self, name,node_map,wire_map)

        if load_dict["Sensitivity_analysis"]["SW"]["name"]:
            name = load_dict["Sensitivity_analysis"]["SW"]["name"]
            position = load_dict["Sensitivity_analysis"]["SW"]["position"]
            bran = load_dict["Sensitivity_analysis"]["SW"]["bran"]

            Strategy.Change_SW().apply(load_dict,name,position,bran)

        if load_dict["Sensitivity_analysis"]["DE"]:
            Strategy.Change_DE_max().apply(self,load_dict["Sensitivity_analysis"]["DE"])
            self.calculate(self.dt, self.H, self.sources)
            pd.DataFrame(self.run_measure()).to_csv("DE_modified.csv")


        if load_dict["Sensitivity_analysis"]["ROD"]:
            for tower in load_dict["Tower"]:
                if tower["name"] ==load_dict["Sensitivity_analysis"]["ROD"]["tower"]:
                    Strategy.Change_ROD().apply(self,load_dict,
                                        load_dict["Sensitivity_analysis"]["ROD"]["lump"],
                                        load_dict["Sensitivity_analysis"]["ROD"]["r"],
                                        load_dict["Sensitivity_analysis"]["ROD"]["l"])

       # self.calculate(self.dt)



