from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
class Strategy(ABC):

    def __init__(self):
        self.capacitance_matrix = None

    @abstractmethod
    def apply(netwotk,dt,Nt):
        C = np.array(netwotk.capacitance_matrix)  # 点点
        G = np.array(netwotk.conductance_matrix)
        L = np.array(netwotk.inductance_matrix)  # 线线
        R = np.array(netwotk.resistance_matrix)
        ima = np.array(netwotk.incidence_matrix_A)  # 线点
        imb = np.array(netwotk.incidence_matrix_B.T)  # 点线


class baseStrategy(Strategy):
    def apply(self,network,dt):

        if not network.switch_disruptive_effect_models and not network.voltage_controled_switchs and not network.time_controled_switchs and not network.nolinear_resistors:
            print("linear calculation is used")
            C = np.array(network.capacitance_matrix)#点点
            G = np.array(network.conductance_matrix)
            L = np.array(network.inductance_matrix)#线线
            R = np.array(network.resistance_matrix)
            ima = np.array(network.incidence_matrix_A)#线点
            imb = np.array(network.incidence_matrix_B.T)#点线
            source = np.array(network.sources)
            nodes = len(network.capacitance_matrix.columns.tolist())
            branches = len(network.inductance_matrix.columns.tolist())
            time_length = len(network.sources.columns.tolist())

            out = np.zeros((branches + nodes, time_length))
            branches, nodes = ima.shape
            for i in range(time_length - 1):
                Vnode = out[:nodes, i].reshape((-1,1))
                Ibran = out[nodes:, i].reshape((-1,1))
                Isource = source[:,i+1].reshape((-1,1))
                LEFT = np.block([[-ima, -R - L / dt], [G + C / dt, -imb]])
                inv_LEFT = np.linalg.inv(LEFT)
                RIGHT = np.block([[(-L / dt).dot(Ibran)], [(C / dt).dot(Vnode)]])
                #temp_result = inv_LEFT.dot(RIGHT)
                temp_result = inv_LEFT.dot(Isource + RIGHT)
                out[:, i + 1] = np.copy(temp_result)[:,0]
            network.solution = pd.DataFrame(out, index=network.capacitance_matrix.columns.tolist()+network.inductance_matrix.columns.tolist())
        else:
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
                Isource = source[:, i+1].reshape((-1, 1))
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


class Monteclarlo(Strategy):
    def apply(self, network):
        print("montecarlo")

