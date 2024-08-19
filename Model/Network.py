import numpy as np

from Model.Cable import Cable
from Model.Lightning import Lightning
from Model.Tower import Tower
from Model.Wires import OHLWire


class Network:
    def __init__(self, tower:Tower, cable:Cable, OHL:OHLWire, lightning:Lightning, H_matrix:np.matrix):
        self.Tower = tower
        self.Cable = cable
        self.OHL = OHL
        self.Lightning = lightning
        self.H_matrix = H_matrix


    def update_H(self, H_matrix:np.matrix):

    def measurement(self,):

