import numpy as np
import pandas as pd
import math
import os

np.seterr(divide="ignore",invalid="ignore")

class Init:
    def Global_Init(FDIR): # 给main_modelfileread用
        # *********** 4 digits for Key Cir, Node and Bran ID and names *************
        # --------------------(1) Global parameters --------------------------------

        IDformat = '%04d'  # 类属性
        NTower = 4  # number of Tower # of towers = x
        NSpan = 3  # number of Span # of spans = x
        NCable = 0  # number of Cable # of cables = x

        dT = 1e-8  # t increment
        Ns = 2000  # calculation steps of each stroke (default
        Nt = Ns  # calculation steps of each stroke
        T0 = 1e-3  # time interval of adjecent strokes
        Tmax = 2e-3  # max time of cacl. for multiple strokes
        slg = 20  # Segment length

        ## derived simulation constants
        Nmax = math.floor(Tmax / dT)
        N0 = math.floor(T0 / dT)
        if Ns > N0:
            Ns = N0

        ## Ground soil parameters
        GND = {
            'gnd': int(2),  # GND mode;: 0 free-space, 1 PGD, 2 LSG
            'gndcha': int(2),  # GND mode;: 0 free-space, 1 PGD, 2 LSG
            'mur': 1,
            'epr': 4,
            'sig': 1e-3,
            'glb': 1  # the same gnd data for whole sys.(0/1=N/Y)
        }

        # Circuit
        Cir_dat = np.array([
            [1001, 1, 1],  # SW:10x; 35kV: 20x; 10kv: 30x; 0.4kv: 40x
            [3001, 3, 1],  # Cir ID + # of each circuit + Ser # on the tower
            [6001, 4, 1]  # underground cable
        ])

        Cir_num = np.array([
            [2, 1, 0, 1, 0, 1],  # gnd cable (not counted in num(1)
            [8, 1, 0, 3, 0, 4]
        ])
        Cir = {'dat': Cir_dat, 'num': Cir_num}

        ## Other constants

        # SSdy = {"flag":0}   # defualt = no sensitivity study

        # collected from a preset conductor table (database)
        VFIT = {
            'fr': np.array([1, 100, 1000]),
            'rc': np.array([1e-3, 1e-4, 1e-5, 1e-5, 1e-5, 1e-5]),  # internal imepdance of conductor
            'dc': np.array([2e-7, 3e-6, 4e-6, 5e-6, 1e-5, 1e-5]),  # order=3, r0+d0*s+sum(ri/s+di)
            'odc': 5,  # VFIT order of conductors
            # calculated within a SPAN module
            'rg': np.array([1e-3, 1e-4, 1e-5, 1e-5]),  # ground impedance
            'dg': np.array([2e-7, 3e-6, 4e-6, 5e-6]),  # order=3, r0+d0*s+sum(ri/s+di)
            'odg': 5  # VFIT order of conductors
        }
        VFIT['fr'] = np.concatenate([
            np.arange(1, 91, 10),
            np.arange(100, 1000, 100),
            np.arange(1000, 10000, 1000),
            np.arange(10000, 100000, 10000),
            np.arange(100000, 800000, 100000)
        ])

        # Define TOW/OHL/CABLE Type parameters
        A = np.array([
            [-1, 1, 0, 0],  # incidence matrix btw. span and tower
            [0, -1, 1, 0],
            [0, 0, -1, 1],
        ])

        # Acab
        if NCable == 1:
            Acab = np.array([
                [-1, 0, 0, 0, 1]
            ])  # underground cable btw T1 and T5
        elif NCable == 0:
            Acab = np.array([
            ])  # underground cable btw T1 and T5
        # nle_flag
        nle_flag = 0  # 0 for linear system; 1 for nonlinear system;
        return {
            "FDIR": FDIR,
            "IDformat": IDformat,
            "NTower": NTower,
            "NSpan": NSpan,
            "NCable": NCable,
            "dT": dT,
            "Ns": Ns,
            "Nt": Nt,
            'T0': T0,
            "Tmax":Tmax,
            "slg": slg,
            "Nmax":Nmax,
            "N0":N0,
            "GND": GND,
            "Cir": Cir,
            # 'SSdy': SSdy,
            "VFIT": VFIT,
            "A": A,
            "Acab": Acab,
            'nle_flag':nle_flag,
            'FOtype':"",
        }

    def LGT_Init(GLB, FDIR):
        #(1) General parameters
        Lch = {
            'dT': GLB['dT'],
            'Nt': GLB['Nt'],
            'mur': GLB['GND']['mur'],
            'sig': GLB['GND']['sig'],
            'eps': GLB['GND']['epr'],
            'gnd': GLB['GND']['gnd'],       # GND mode;: 0 free-space, 1 PGD, 2 LSG
            'gndcha': GLB['GND']['gndcha']  # GND mode;: 0 free-space, 1 PGD, 2 LSG
        }

        #(2) Channel model---- 1=Nc,2=dH,3=H0,4=flag_type,5=H,6=lamda,7=vco
        channel_model = 'MTLE'  # Chan model (MTLE model(lamda,vcof))
        channel_filename = "Channel Model Table.xlsx"

        channel_data = None
        channel_model_id = None
        path = FDIR['ecal']
        excel_path = path + '/' + channel_filename

        # reading table
        raw_data = pd.read_excel(excel_path, header=None).values

        for row in raw_data:
            if row[0] == channel_model:
                channel_data = row[1:-1]
                channel_model_id = row[-1]
                break

        Lch['flg'] = channel_model_id   # 1=TL model,2=MTLL model(H),3=MTLE model
        Lch['H'] = channel_data[0]      # channel attenuation coef
        Lch['lam'] = channel_data[1]
        Lch['vcf'] = channel_data[2]

        Lch['H0'] = 1000    # Channel height
        Lch['dH'] = 10      # Length of channel segments
        Lch['Nc'] = Lch['H0'] / Lch['dH']  # Number of channel segments
        Lch['pos'] = []     # ind: (x,y,0), dir: (S/C/T ID, seg ID, phase, seg)

        # Generate LGT data
        LGT = {
            'Lch': Lch,
            'Soc': []
        }
        return LGT

    def Simulation_LGT_Init(GLB, LGT):
        """
        Getting Source.Posi/data and SSdy for LGT Simulation via UI/input files

        Args:
        - GLB (dict): Global parameters
        - LGT (dict): Lightning simulation parameters

        Returns:
        - GLB (dict): Updated Global parameters
        - LGT (dict): Updated Lightning simulation parameters
        """
        # (0) Initial Constants
        dT = GLB['dT']
        T0 = GLB['T0']  # stroke interval of 1ms
        Nt = GLB['Nt']  # sample# for each stroke (cal)
        N0 = GLB['N0']  # sample# for each stroke (max)
        Nmax = GLB['Nmax']  # max sample #
        Lch = LGT['Lch']

        # (1) Getting sensitivity study table (SSdy)
        # [NoSSdy, FO(l/2) NonR(1/2/..),SA file name, Gnd(1/2), Sig(1/2/3/4
        flag = [0, 0, 0, 0, 0, 4, 0]
        SSdy = {'flag': flag}  # 0="diable", >0="enable", Sig=1(tower
        SSdy['Nca'] = 2  # # of cases
        # SSdy['dat'] = [100, 200]  # CFO=100kV, 200kV
        SSdy['dat'] = [[0.02, 10], [0.002, 4]]  # soil conductivity
        # SSdy['dat'] = ["Model_GRID3.xlsx", "Model_GRID3b.xlsx"]  # ground grid
        # SSdy['dat'] = ["Model_SARR1.xlsx", "Model_SARR2.xlsx"]  # SA model
        # SSdy['dat'] = ["SWH01", "SWH02"]  # CFO=100kV, 200kV

        # (2) Getting current waveform considering a multiple-stroke flash
        flash = {'head': [1, 2, 2, 0],  # 2 strokes with Heidler model
                 'flag': [1, 1],  # Strokes are enabled in simulation
                 'para': [],  # Current parameters (e.g., 10/350us)
                 'wave': [[10e4, 0.939, 1, 25, 10],  # wave patameters (kA/us) of 1/50us
                          [10e4, 0.939, 1, 25, 10]]}  # wave patameters (kA/us) of 1/50us

        # icur = np.zeros((Nmax, 1))  # current up to Nmax
        # for ik in range(flash['head'][1]):  # loop for stroke #
        #     if flash['flag'][ik] == 1:  # enable for stroke i
        #         out = Source_Current_Generator(flash, T0, N0, Nt, dT)
        #         tmpdex = (ik - 1) * N0 + np.arange(Nt)
        #         icur[tmpdex, 0] = out  # add all stroke currents with zeros
        Icur = Source_Current_Generator(flash, T0, N0, dT)  # flash current (mul-s)

        # (3) Getting soc.typ/pos/dat
        Soc = {'typ': 0,  # 1= direct stroke, 0= indirect
               'pos': [0, 500, 500, 0, 0, 0],  # (Ind) 1st span, SW1, Seg3
               'dat': Icur,  # flash current (mul-s)
               'flash': flash}  # multiple stroke data

        Lch['curr'] = []  # channel stroke current
        Lch['pos'] = Soc['pos'][1:3]  # channel position

        # (4) Summary info
        LGT['Soc'] = Soc
        LGT['SSdy'] = SSdy
        LGT['Lch'] = Lch
        GLB['Soc'] = Soc
        GLB['SSdy'] = SSdy

        return GLB, LGT

def Source_Current_Generator(flash, T0, N0, dT):
        """
        Generate current waveform data

        Args:
        - flash (dict): Flash parameters
        - T0 (float): Stroke interval of 1ms
        - N0 (int): Total output points
        - dT (float): Step size of time

        Returns:
        - icur (ndarray): Current waveform data
        """
        NumStr = flash['head'][1]  # Number of strokes
        WavMod = flash['head'][2]  # Waveform model
        inputdata_flag = flash['head'][3]  # 1=Yes, 0=No

        icur = np.array([])
        for i in range(NumStr):
            wpara = flash['wave'][i]  # Current parameters of a flash
            if inputdata_flag == 1:
                print("to be developed")  # data from input files
            else:
                if WavMod == 2:  # Heidler model
                    ist = Heidler_Waveform_Generator(wpara, N0, dT)
                elif WavMod == 1:  # CIGRE model
                    ist = CIGRE_Waveform_Generator(wpara, N0, dT)

            # Add cos(theta) window at the tail
            Ntail = int(np.floor((T0 / 10) / dT))  # of points for the tail with window
            theta = np.arange(Ntail) * np.pi / Ntail
            coswin = 0.5 * np.cos(theta) + 0.5 # [1 0];
            ist[-Ntail:] = ist[-Ntail:] * coswin

            icur = np.concatenate([icur, ist])

        return icur

def Heidler_Waveform_Generator(para, N0, dT):
    """
    Function:       sr_herdler
    Description:    Lightning Source Generation using HERDLER Function.

    Calls:

     Input:          Amp  --  amplitude of the lightning source
                 Tf   --  the front duration in [s]. Intercal between t=0 to the time
                          of the function peak.
                 tau  --  the stoke duration in [s]. Interval between t=0 and the
                          point on the tail whrer the function amplitude has fallen
                          to 37% of its peak value.
                 n    --  factor influencing the rate of rise of the function.
                          Increased n increases the maximum steepnes.
                 Tmax --  ending time in [us].
                 dt   --  time interval [us]
     Output:         ist  --  output of the source ( U or I )
                     t    --  the time sequence (us)
     Others:
     Author:         Chen Hongcai
     Email :         hc.chen@live.com
     Date:           2013-12-16
                     2024-2-25 by YD
    """
    amp = para[0]
    k = para[1]
    tau1 = para[2] * 1e-6
    tau2 = para[3] * 1e-6
    n = para[4]

    tus = np.arange(0, N0) * dT
    ist = (amp / k) * (tus / tau1) ** n / (1 + (tus / tau1) ** n) * np.exp(-tus / tau2)

    return ist

def CIGRE_Waveform_Generator(WavePara, N0, dT):
    """
    Iout = ((t<= tn).* (A*t+B*(t^n)) + (t>tn) .* (I1*exp(-(t-tn)/t1) - I2*exp(-(t-tn)/t2)))*(Ipi/Ipc);
    """
    tn = WavePara[0]
    A = WavePara[1]
    B = WavePara[2]
    n = WavePara[3]
    I1 = WavePara[4]
    t1 = WavePara[5]
    I2 = WavePara[6]
    t2 = WavePara[7]
    Ipi = WavePara[8]
    Ipc = WavePara[9]

    t = np.arange(0, N0) * dT
    T0_le_tn = t[t <= tn]  # index for t=<tn
    Iout1 = A * T0_le_tn + B * (T0_le_tn ** n)
    T0_gt_tn = t[t > tn]   # index for t=<tn
    Iout2 = I1 * np.exp(-(T0_gt_tn - tn) / t1) - I2 * np.exp(-(T0_gt_tn - tn) / t2)

    Iout = np.concatenate([Iout1, Iout2]) * (Ipi / Ipc)

    return Iout
