import numpy as np
import os

from Tower_V3.Block_Circuit_Build.Tower_Circuit_Build import Tower_Circuit_Build
from Tower_V3.Block_Circuit_Build.Span_Circuit_Build import Span_Circuit_Build
from Tower_V3.Block_Circuit_Build.Cable_Circuit_Build import Cable_Circuit_Build

np.seterr(divide="ignore",invalid="ignore")


##
# Read input files of Tower/Span/Cable
# Build Circuit Models of Tower/Span/Cable
# including Mapping tables, but not the lightning source

class Block_Circuit_Build():
    def __init__(self, FDIR):
        self.FDIR = FDIR  # Initialize the Block_Circuit_Build class with the given file directory paths

    def Block_Circuit_Build(self, GLB): # Build tower, span, and cable circuit models
        # define table path
        datafiles_path = self.FDIR['datafiles'] # Set the path to the data files directory
        VFIT = GLB['VFIT'] # Get the VFIT value from the GLB dictionary

        # (0) Init Tower, Span, and Cable lists
        Tower = []
        Span = []
        Cable = []

        TowerCircuitBuild = Tower_Circuit_Build(datafiles_path)  # Create an instance of Tower_Circuit_Build class
        SpanCircuitBuild = Span_Circuit_Build(datafiles_path)
        CableCircuitBuild = Cable_Circuit_Build(datafiles_path)  # Construct the Cable model file name

        # (1) Build tower circuit models
        for i in range(1, GLB['NTower'] + 1):
            str1 = "Input_Tower" # Define the prefix of the Tower model file name
            str2 = ".xlsx" # Define the file extension
            TowerModelName = str1 + str(i) + str2   # Construct the Tower model file name
            Tower_Model = TowerCircuitBuild.Tower_Circuit_Build(TowerModelName, VFIT, GLB)
            # (has be done) update CK_Para.P by inclduing Pa and Pg
            Tower.append(Tower_Model)  # append to Tower list

        # (2) Build span circuit models
        nrow, ncol = GLB['A'].shape # Get the dimensions of the GLB['A'] array
        TCom = {'head':{}, 'tail':{}} # Initialize the TCom dictionary

        for i in range(1, GLB['NSpan'] + 1):
            for j in range(ncol):
                if GLB['A'][i - 1, j] == -1:
                    hid = j         # Tower Node
                elif GLB['A'][i - 1, j] == 1:
                    tid = j         # Tower Node

            # Access contents of cells of a string array by x[cell id][0][string id]
            TCom['head']['list'] = np.array(["X01", "X02", "X03", "X04"])  # SW + A/B/C
            TCom['tail']['list'] = np.array(["X01", "X02", "X03", "X04"])  # SW + A/B/C
            if i == 4 or i == 5:
                TCom['head']['list'] = np.array(["X01", "X15", "X16", "X17"])

            str1 = "Input_Span"
            str2 = ".xlsx"
            SpanModelName = str1 + str(i) + str2 # Construct the Span model file name
            Span_Model, S2Tmap = SpanCircuitBuild.Span_Circuit_Build(SpanModelName, TCom, Tower[hid]['Info'], Tower[hid]['Node'],
                                              Tower[tid]['Info'], Tower[tid]['Node'], VFIT, GLB)
            Span.append(Span_Model) # append to Span list

            # Generate mapping tables of Towers from Span
            Tower[hid]['T2Smap'], Tower[tid]['T2Smap'] = TowerCircuitBuild.Tower_Map_Update(Tower[hid]['T2Smap'], Tower[tid]['T2Smap'], S2Tmap)
            # (to be done) update CK_Para.Cw (Cw oder = T2Smap order of line)

        # (3) Build Cable circuit models
        if GLB['Acab']: # Check if GLB['ACAB'] is not empty
            nrow, ncol = GLB['Acab'].shape

        for i in range(1, GLB['NCable'] + 1):
            for j in range(ncol):
                if GLB['Acab'][i - 1, j] == -1:
                    hid = j  # Tower Node
                elif GLB['Acab'][i - 1, j] == 1:
                    tid = j  # Tower Node

            TCom['head']['list'] = ["X34", "X35", "X36", "X37"]  # AM + A/B/C
            TCom['tail']['list'] = ["X34", "X35", "X36", "X37"]  # AM + A/B/C

            str1 = "Input_Cable"
            str2 = ".xlsx"

            CableModelName = str1 + str(i) + str2
            # Create an instance

            Cable_Model, C2Tmap = CableCircuitBuild.Cable_Circuit_Build(CableModelName, TCom,
                                Tower[hid]['Info'], Tower[hid]['Node'],
                                Tower[tid]['Info'], Tower[tid]['Node'], VFIT, GLB)
            Cable.append(Cable_Model)

            # Generate mapping tables of Towers from Cable
            Tower[hid]['T2Cmap'], Tower[tid]['T2Cmap'] = TowerCircuitBuild.Tower_Map_Update(Tower[hid]['T2Cmap'], Tower[tid]['T2Cmap'], C2Tmap)
        else:
            pass

        # (4) Obtain Connection data Cw
        for i in range(GLB['NTower']):
            CK_Para = Tower[i]['CK_Para']
            CK_Para['Cw'], CK_Para['Cwc'] = TowerCircuitBuild.Tower_Line_Connect(Tower[i], Span, Cable)
            Tower[i]['CK_Para'] = CK_Para

        return Tower, Span, Cable