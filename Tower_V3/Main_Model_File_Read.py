import numpy as np
import os

np.seterr(divide="ignore",invalid="ignore")

from Initialization import Init
from Tower_V3.Simulation_LGT.Simulation_LGT import Simulation_LGT
from Tower_V3.Simulation_MCLGT.Simulation_MC_LGT import Simulation_MC_LGT
from Tower_V3.Block_Circuit_Build.Block_Circuit_Build import Block_Circuit_Build



## set the global path for program
CurDir = os.getcwd() # Get the current working directory

## same with Tower_V9d
FDIR = {} # set the file directory path
FDIR['main'] = CurDir  # Main directory
FDIR['vfit'] = os.path.join(CurDir, 'PROJ_VFIT')  # Vector fitting directory
FDIR['lcal'] = os.path.join(CurDir, 'PARA_LCAL')  # Parameter calibration directory
FDIR['ecal'] = os.path.join(CurDir, 'PARA_ECAL')  # Energy calibration directory
FDIR['ecal_vfit'] = os.path.join(CurDir, 'PARA_ECAL', 'Vectorfitting')  # Energy calibration vector fitting directory
FDIR['mclg'] = os.path.join(CurDir, 'PARA_MCLG')  # Multi-circuit current transformer directory
FDIR['mclgoths'] = os.path.join(CurDir, 'PARA_MCLG', 'OTHS')  # Multi-circuit current transformer others directory
FDIR['srcg'] = os.path.join(CurDir, 'PARA_SRCG')  # External power supply parameter directory
FDIR['buld'] = os.path.join(CurDir, 'PARA_BULD')  # Building parameter directory
FDIR['solv'] = os.path.join(CurDir, 'EQUT_SOLV')  # Equation solving directory
# FDIR['dataInputFile'] = input('******* Open Project Folder *********：')  # Project data folder/ not used now
FDIR['LGTSim'] = os.path.join(CurDir, 'Simulation_LGT')  # Lightning surge simulation folder
FDIR['MCLGTSim'] = os.path.join(CurDir, 'Simulation_MCLGT')  # Multi-circuit current transformer simulation folder

## former
FDIR['mcls'] = os.path.join(CurDir, 'PARA_MCLS')  # Multi-circuit current transformer directory
FDIR['dataMCLG'] = os.path.join(CurDir, 'DATA_MCLG') # Data - Multi-circuit current transformer directory
FDIR['dataMCLG2'] = os.path.join(CurDir, 'DATA_MCLG', 'DATA_MCLG_S')
FDIR['dataTempFile'] = os.path.join(CurDir, 'DATA_TempFile')  # Data - Temporary file directory
FDIR['datafiles'] = os.path.join(CurDir, 'DATA_Files')  # Data files directory
FDIR['datafileread'] = os.path.join(CurDir, 'Data_Read')  # Data read directory
FDIR['MC_lightning_data'] = os.path.join(CurDir, 'MC_lightning_data') # MC lightning data directory



## Init for GLB and LGT
GLB = Init.Global_Init(FDIR)
LGT = Init.LGT_Init(GLB, FDIR)

## Build all the circuit information
Block_Circuit_Build = Block_Circuit_Build(FDIR)
Tower, Span, Cable = Block_Circuit_Build.Block_Circuit_Build(GLB)

print("end")

## Perform lightning surge simulation including sensitivty study
# Simulation_LGT = Simulation_LGT(FDIR)
# Output, Tower, Span = Simulation_LGT.Simulation_LGT(Tower, Span, Cable, GLB, LGT)


Simulation_MC_LGT = Simulation_MC_LGT(FDIR)
Output, Tower, Span = Simulation_MC_LGT.Simulation_MC_LGT(Tower, Span, Cable, GLB, LGT)
