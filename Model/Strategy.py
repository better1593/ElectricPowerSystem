from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from Driver.modeling.tower_modeling import tower_building
from Driver.initialization.initialization import initial_device,initial_lump
import json
import pickle
from Model import Network
class Strategy(ABC):

    def __init__(self):
        self.capacitance_matrix = None

    @abstractmethod
    def apply(self,netwotk,dt,Nt):
        C = np.array(netwotk.capacitance_matrix)  # 点点
        G = np.array(netwotk.conductance_matrix)
        L = np.array(netwotk.inductance_matrix)  # 线线
        R = np.array(netwotk.resistance_matrix)
        ima = np.array(netwotk.incidence_matrix_A)  # 线点
        imb = np.array(netwotk.incidence_matrix_B.T)  # 点线

class Linear(Strategy):
    def apply(self,network,dt):
        print("linear calculation is used")
        C = np.array(network.capacitance_matrix)  # 点点
        G = np.array(network.conductance_matrix)
        L = np.array(network.inductance_matrix)  # 线线
        R = np.array(network.resistance_matrix)
        ima = np.array(network.incidence_matrix_A)  # 线点
        imb = np.array(network.incidence_matrix_B.T)  # 点线
        source = np.array(network.sources)
        nodes = len(network.capacitance_matrix.columns.tolist())
        branches = len(network.inductance_matrix.columns.tolist())
        time_length = len(network.sources.columns.tolist())

        out = np.zeros((branches + nodes, time_length))
        branches, nodes = ima.shape
        for i in range(time_length - 1):
            Vnode = out[:nodes, i].reshape((-1, 1))
            Ibran = out[nodes:, i].reshape((-1, 1))
            Isource = source[:, i + 1].reshape((-1, 1))
            LEFT = np.block([[-ima, -R - L / dt], [G + C / dt, -imb]])
            inv_LEFT = np.linalg.inv(LEFT)
            RIGHT = np.block([[(-L / dt).dot(Ibran)], [(C / dt).dot(Vnode)]])
            # temp_result = inv_LEFT.dot(RIGHT)
            temp_result = inv_LEFT.dot(Isource + RIGHT)
            out[:, i + 1] = np.copy(temp_result)[:, 0]
        network.solution = pd.DataFrame(out,
                                        index=network.capacitance_matrix.columns.tolist() + network.inductance_matrix.columns.tolist())

class Change_light_pos(Strategy):
    def apply(self,network,lightning,new_pos):
        print("change light position calculation is used")

        lightning.channel.hit_pos = new_pos["position"]
        network.source_calculate(lightning, new_pos)

class Change_light_waveform(Strategy):
    def apply(self, network, lightning, new_waveform, pos_dict):
        print("change light waveform calculation is used")

        lightning.channel.hit_pos = new_waveform
        network.source_calculate(lightning, pos_dict)

class Change_light_parameters(Strategy):
    def apply(self, network, lightning, new_parameters, pos_dict):
        print("change light parameters calculation is used")
        for stroke in lightning.strokes:
            stroke.parameters = new_parameters #
            stroke.current_waveform = []
            stroke.calculate()
        network.source_calculate(lightning, pos_dict)

class Change_Arrestor_pos(Strategy):
    def apply(self, network,name, wire_mapping,node_mapping):
        print("changing arrestor position calculation")
        for i,tower in enumerate(network.towers):
            for j,device in enumerate(tower.devices):
                for index,arrestor in enumerate(device.arrestors):
                    if arrestor.name == name:
                        network.switch_disruptive_effect_models.remove(device.switch_disruptive_effect_models)
                        network.voltage_controled_switchs.remove(device.voltage_controled_switchs)
                        network.time_controled_switchs.remove(device.time_controled_switchs)
                        network.nolinear_resistors.remove(device.nolinear_resistors)

                        arrestor.capacitance_matrix = arrestor.capacitance_matrix.rename(columns=node_mapping,index=node_mapping)
                        arrestor.inductance_matrix = arrestor.inductance_matrix.rename(columns=wire_mapping,index=wire_mapping)
                        arrestor.incidence_matrix_A = arrestor.incidence_matrix_A.rename(columns=node_mapping,index=wire_mapping)
                        arrestor.inductance_matrix_B = arrestor.inductance_matrix_B.rename(columns=node_mapping,index=wire_mapping)
                        arrestor.resistance_matrix = arrestor.resistance_matrix.rename(columns=wire_mapping,index=wire_mapping)
                        arrestor.conductance_matrix = arrestor.conductance_matrix.rename(columns=node_mapping,index=node_mapping)
                        device[index] = arrestor
                tower.devices[j] = device
            tower.reset_matrix()
            gnd = network.ground if network.global_ground == 1 else tower.ground
            tower_building(tower, network.f0, network.max_length, gnd, network.varied_frequency)
            network.switch_disruptive_effect_models.extend(device.switch_disruptive_effect_models)
            network.voltage_controled_switchs.extend(device.voltage_controled_switchs)
            network.time_controled_switchs.extend(device.time_controled_switchs)
            network.nolinear_resistors.extend(device.nolinear_resistors)

            network.towers[i] = tower
            network.combine_parameter_matrix()
            network.calculate(network.dt)



class Change_ROD(Strategy):
    def apply(self, network, lump_dict,name,r,l, dt):
        print("change ROD calculation is used")
        for i,tower in enumerate(network.towers):
            for lump in lump_dict:
                if lump.name == name:
                    network.switch_disruptive_effect_models.remove(lump.switch_disruptive_effect_models)
                    network.voltage_controled_switchs.remove(lump.voltage_controled_switchs)
                    network.time_controled_switchs.remove(lump.time_controled_switchs)
                    network.nolinear_resistors.remove(lump.nolinear_resistors)


                    lump_dict.value1 = r
                    lump_dict.value2 = l
                    lumps,measurement = initial_lump(lump_dict, network.dt, network.T,network.measurement)
                    tower.lumps = lumps

                    network.switch_disruptive_effect_models.extend(lump.switch_disruptive_effect_models)
                    network.voltage_controled_switchs.extend(lump.voltage_controled_switchs)
                    network.time_controled_switchs.extend(lump.time_controled_switchs)
                    network.nolinear_resistors.extend(lump.nolinear_resistors)
            network.towers[i] = tower
            tower.reset_matrix()
            gnd = network.ground if network.global_ground == 1 else tower.ground
            tower_building(tower, network.f0, network.max_length, gnd, network.varied_frequency)

            network.combine_parameter_matrix()
            network.calculate(network.dt)

class Change_ground(Strategy):
    def apply(self,  load_dict, new_parameter):
        print("change ground calculation")
        load_dict['Global']["ground"] = new_parameter  # 修改值

        with open("modified", 'w') as file:
            json.dump(load_dict, file, indent=4)

        network = Network()
        network.run(load_dict)
class Change_SW(Strategy):
    def apply(self,  load_dict, name,position,bran):

        for index,ohl in enumerate(load_dict['OHL']):
            if ohl.name == name:
                if position:
                    ohl["position"] = position # 修改值
                if bran:
                    ohl["bran"] = bran # 修改值
                load_dict['OHL'][index] = ohl

        with open("modified", 'w') as file:
            json.dump(load_dict, file, indent=4)

        network = Network()
        network.run(load_dict)
class NonLinear(Strategy):
    def apply(self,network,dt):
        print("Nonlinear calculation is used")
        branches, nodes = network.incidence_matrix_A.shape
        source = np.array(network.sources)
        time_length = len(network.sources.columns.tolist())
        out = np.zeros((branches + nodes, time_length))
        # source = np.array(sources)
        for i in range(time_length - 1):
            C = network.capacitance_matrix.to_numpy()  # 点点
            G = network.conductance_matrix.to_numpy()
            L = network.inductance_matrix.to_numpy()  # 线线
            R = network.resistance_matrix.to_numpy()
            ima = network.incidence_matrix_A.to_numpy()  # 线点
            imb = network.incidence_matrix_B.T.to_numpy()  # 点线
            Vnode = out[:nodes, i].reshape((-1, 1))
            Ibran = out[nodes:, i].reshape((-1, 1))
            Isource = source[:, i + 1].reshape((-1, 1))
            LEFT = np.block([[-ima, -R - L / dt], [G + C / dt, -imb]])
            inv_LEFT = np.linalg.inv(LEFT)
            RIGHT = np.block([[(-L / dt).dot(Ibran)], [(C / dt).dot(Vnode)]])

            temp_result = inv_LEFT.dot(Isource + RIGHT)
            # temp_result = inv_LEFT.dot(RIGHT)
            out[:, i + 1] = np.copy(temp_result)[:, 0]
            temp_result = pd.DataFrame(temp_result,
                                       index=network.capacitance_matrix.columns.tolist() + network.inductance_matrix.columns.tolist())

            t = dt * (i + 1)
            network.update_H(temp_result, t)

        network.solution = pd.DataFrame(out,
                                        index=network.capacitance_matrix.columns.tolist() + network.inductance_matrix.columns.tolist())


class Change_DE_max(Strategy):
    def apply(self,network,DE_max):
        print("Changing DE max calculation is used")
        network.reset_matrix()
        for index,lump in enumerate(network.switch_disruptive_effect_models):
            lump.parameters['DE_max'] = DE_max
            network.switch_disruptive_effect_models[index] = lump

        network.calculate()
        with open('output_change_DE.pkl', 'wb') as f:
            pickle.dump(network.run_measure(), f)



class Measurement(Strategy):
    def apply(self,measurement,solution,dt):
        results = {}
        for key, value in measurement.items():
            data_type = value[0]  # 0:'branch',1: 'normal lump'
            measurement_type = value[1]  # 1:'current',2:'voltage'
            # 处理支路名或节点名
            if data_type == 0:
                branch_name = key
                current = solution.loc[branch_name].tolist() if branch_name in solution.index else None
                node1 = solution.loc[value[3]].tolist() if value[3] in solution.index else None
                node2 = solution.loc[value[4]].tolist() if value[4] in solution.index else None
                voltage = [abs(a - b) for a, b in zip(node1, node2)] if node1 and node2 else None
                p = [a * b for a, b in zip(current, voltage)] if current and voltage else None
                E = sum([i * dt for i in p]) if p else None
                if measurement_type == 1:
                    results["current"] = current
                elif measurement_type == 2:
                    results["voltage"] = voltage
                elif measurement_type == 3:
                    results["P"] = p
                elif measurement_type == 4:
                    results["E"] = E
                    results["P"] = p
                    results["voltage"] = voltage
                    results["current"] = current
                elif measurement_type == 11:
                    results["E"] = E

                results["index"] = [key,value[3],value[4]]
            elif data_type == 1:
                lump_name = key
                # 处理支路名可能是列表的情况
                branches = value[3]
                dict_result = {}
                for bran,n1,n2 in zip(value[2],value[3],value[4]):
                    current = solution.loc[bran].tolist() if bran in solution.index else None
                    node1 = solution.loc[n1].tolist() if n1 in solution.index else None
                    node2 = solution.loc[n2].tolist() if n2 in solution.index else None
                    voltage = [abs(a - b) for a, b in zip(node1, node2)] if (node1 and node2)  else None
                    p = [a * b for a, b in zip(current, voltage)] if (current and voltage)  else None
                    E = sum([i*dt for i in p ]) if p else None

                    if measurement_type == 1:
                        dict_result["current"] = current
                    elif measurement_type == 2:
                        dict_result["voltage"] = voltage
                    elif measurement_type == 3:
                        dict_result["P"] = p
                    elif measurement_type == 4:
                        dict_result["current"] = current
                        dict_result["voltage"] = voltage
                        dict_result["P"] = p
                        dict_result["E"] = E
                    elif measurement_type == 11:
                        dict_result["E"] =E
                dict_result["index"] = [bran,n1,n2]
                results[lump_name] = dict_result
        return results

class Monteclarlo(Strategy):
    def apply(self, network):
        print("montecarlo")

