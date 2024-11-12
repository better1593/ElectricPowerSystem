import numpy as np

from Tower_V3.Initialization import Init
from Tower_V3.Block_Circuit_Build.Tower_Circuit_Build import Tower_Circuit_Build

from Tower_V3.Simulation_LGT.Ground_Potential import Ground_Potential
from Tower_V3.Simulation_LGT.Span_Circuit_Para import Span_Circuit_Para
from Tower_V3.Simulation_LGT.Nle_gen import Nle_gen
from Tower_V3.Simulation_LGT.Solution_S2_Nonlinear import Solution_S2_Nonlinear
from Tower_V3.Simulation_LGT.Mea_out import Mea_out

class Simulation_LGT():
    def __init__(self, FDIR):
        self.FDIR = FDIR  # Initialize the Simulation_LGT class with the given file directory paths

    def Simulation_LGT(self, Tower, Span, Cable, GLB, LGT):
        """
        Perform lightning simulation including sensitivity study, without Monte Carlo analysis

        Args:
        - Tower (dict): Tower data
        - Span (dict): Span data
        - Cable (dict): Cable data
        - GLB (dict): Global parameters
        - LGT (dict): Lightning simulation parameters

        Returns:
        - output (dict): Simulation output data
        - Tower (dict): Updated Tower data
        - Span (dict): Updated Span data
        """
        # Sensitivity study type
        # 0) No sensitivity study
        # 1) Insulation strength
        # 2) nonlinear curve
        # 3) SA device
        # 4) Grounding impedance
        # 5) Soil conductivity
        # 6) Lightning source parameters (position, current magnitude/waveform)

        # (1) Getting source information of LGT Simulation Module via UI/input file
        GLB, LGT = Init.Simulation_LGT_Init(GLB, LGT)

        # (2) Getting FO index selection
        FO_Type = 1  # tradiitonal definition
        GLB['FOtype'] = FO_Type

        # (3) Performing LGT Simulation and getting the output data
        Output, Tower, Span = self.LGT1_Solu(Tower, Span, Cable, GLB, LGT)

        return Output, Tower, Span

    def LGT1_Solu(self, Tower, Span, Cable, GLB, LGT):
        """
        Build matrix equation Ax=b, and Solve the equation for x including:
        (1) Updating T/S/C.Soc
        (2) Updating Sys.A
        (3) Updating Sys.Soc
        (4) Updating Sys.b
        (5) Solving Ax=b
        (6) Getting FO indices with sensitivity study options

        Args:
        - Tower (dict): Tower information
        - Span (dict): Span information
        - Cable (dict): Cable information
        - GLB (dict): Global parameters
        - LGT (dict): Lightning parameters

        Returns:
        - output (list): Output data
        - Tower (dict): Updated Tower information
        - Span (dict): Updated Span information
        """
        # (0) Initialization
        SSdy = GLB['SSdy']
        SSdy_id = np.where(SSdy['flag'] > 0)[0]
        if len(SSdy_id) > 1:
            print('Error in setting sensitivity study Mode (Sim_LGT_Init)')
            return 0, Tower, Span
        TowerCircuitBuild = Tower_Circuit_Build(self.FDIR)  # Create an instance of Tower_Circuit_Build class

        # (1) Initialization with sensitivity study
        Str = ["Single Event", "CFO", "NLE", "SA", "Gnd Grid", "Soil Resistivity", "Curr Para"]
        print("******  Performing Sensitivity Study  -- %s Model  ******\n\n" % Str[SSdy_id])

        FDIR = GLB['FDIR']
        Tfold = FDIR['dataTempFile']

        Icur = LGT['Soc']['dat']  # flash current waveform
        flash = LGT['Soc']['flash']
        StrTyp = LGT['Soc']['typ']  # Dir/ind lightning
        FshID = flash['head'][0]  # flash id
        StrNum = flash['head'][1]  # stroke number
        nst = GLB['N0']  # samples of last stroke

        output = []
        for _ in range(SSdy['Nca']):
            output.append({'Vnt': [], 'Ibt': []})

        # (2) Main loops for multiple strokes --> sensitivity cases
        for k in range(1, StrNum + 1):
            tmpdex = slice((k - 1) * GLB['N0'], (k - 1) * GLB['N0'] + min(GLB['Nt'], nst))
            n0 = min(GLB['Nt'], nst)
            LGT['Lch']['curr'] = Icur[tmpdex]  # update stroke source I

            if SSdy['flag'][5] != 3 and SSdy['flag'][5] != 4:
                Tower, Span, Cable, GLB, LGT = self.LGT_Source_Build(Tower, Span, Cable, GLB, LGT, Tfold)

            # Loop for sensitivity cases for every stroke
            for i in range(1, SSdy['Nca'] + 1):
                if SSdy['flag'][0] > 0:
                    print(f'Case {i} (Flash ID = {FshID}, Lightning Type = {StrTyp}, Stroke ID = {k}): '
                          f'Single-event Lightning Surge Analysis')
                if SSdy['flag'][1] > 0:
                    for j in range(GLB['NTower']):
                        Swh = Tower[j]['CK_Para']['Swh']
                        nrow = Swh['dat'].shape[0]
                        Swh['dat'][:nrow, 0] = SSdy['dat'][i - 1]
                        Tower[j]['CK_Para']['Swh'] = Swh
                    print(f'Case {i} (Flash ID = {FshID}, Lightning Type = {StrTyp}, Stroke ID = {k}): '
                          f'CFO = {SSdy["dat"][i - 1]}')
                if SSdy['flag'][2] > 0:
                    for j in range(GLB['NTower']):
                        Nle = Tower[j]['CK_Para']['Nle']
                        nrow = Nle['dat'].shape[0]
                        Nle['dat'][:nrow, 0] = SSdy['dat'][i - 1]
                        Tower[j]['CK_Para']['Nle'] = Nle
                    print(f'Case {i} (Flash ID = {FshID}, Lightning Type = {StrTyp}, Stroke ID = {k}): '
                          f'NLE = {SSdy["dat"][i - 1]}')
                if SSdy['flag'][3] > 0:
                    for j in range(GLB['NTower']):
                        Blok = Tower[j]['Blok']
                        if not Blok['sar']:  # No Sa provided
                            continue
                        n0 = Blok['sar']['list'].shape[0]
                        if n0 != 6:
                            print('error of SA terminal number in LGT1_Solu')
                            return
                        Tower0 = Tower[j]['Tower0']
                        Blok['name'][2] = SSdy['dat'][i - 1]  # SA name
                        Tower0['Blok'] = Blok
                        Tower[j]['Tower0'] = Tower0
                        Tower[j] = Tower_Circuit_Build.Tower_Circuit_Update(Tower[j])  # rebuild Tower Model
                    print(f'Case {i} (Flash ID = {FshID}, Lightning Type = {StrTyp}, Stroke ID = {k}): '
                          f'SA Model = {SSdy["dat"][i - 1]}')
                if SSdy['flag'][4] > 0:
                    for j in range(GLB['NTower']):
                        Blok = Tower[j]['Blok']
                        if not Blok['grd']:
                            continue
                        n0 = Blok['grd']['list'].shape[0]
                        if n0 > 1:
                            print('error of grounding grid number in LGT1_Solu')
                            return
                        Tower0 = Tower[j]['Tower0']
                        Blok['name'][4] = SSdy['dat'][i - 1]  # ground grid name
                        Tower0['Blok'] = Blok
                        Tower[j]['Tower0'] = Tower0
                        Tower[j] = Tower_Circuit_Build.Tower_Circuit_Update(Tower[j])  # rebuild Tower Model
                    print(f'Case {i} (Flash ID = {FshID}, Lightning Type = {StrTyp}, Stroke ID = {k}): '
                          f'Grounding Grid Model = {SSdy["dat"][i - 1]}')
                if SSdy['flag'][5] > 0:
                    index = 1
                    Tfold = FDIR['dataTempFile']
                    if SSdy['flag'][5] in [1, 2, 3, 4]:
                        for j in range(GLB['NTower']):
                            GND = Tower[j]['GND']
                            GND['sig'] = SSdy['dat'][i - 1, 0]
                            GND['epr'] = SSdy['dat'][i - 1, 1]
                            Tower[j]['GND'] = GND

                            Node = Tower[j]['Node']
                            CK_Para = Tower[j]['CK_Para']
                            CK_Para['G'], CK_Para['P'] = Ground_Potential(CK_Para, Node, GND)
                            Tower[j]['CK_Para'] = CK_Para

                            Tower[j] = self.Tower_Circuit_Source(Tower[j], GLB, LGT, index, Tfold)
                    if SSdy['flag'][5] in [2, 3, 4]:
                        for j in range(GLB['NSpan']):
                            GND = Span[j]['GND']
                            GND['sig'] = SSdy['dat'][i - 1, 0]
                            GND['epr'] = SSdy['dat'][i - 1, 1]
                            Span[j]['GND'] = GND

                            Span[j] = self.Span_Circuit_Source(Span[j], GLB, LGT, index, Tfold)
                    if SSdy['flag'][5] == 4:
                        for j in range(GLB['NSpan']):
                            GND = Span[j]['GND']
                            GND['sig'] = SSdy['dat'][i - 1, 0]
                            GND['epr'] = SSdy['dat'][i - 1, 1]
                            Span[j]['GND'] = GND

                            Span[j]['Para'] = Span_Circuit_Para(Span[j]['Info'], Span[j]['OHLP'], Span[j]['GND'])
                    print(f'Case {i} (Flash ID = {FshID}, Lightning Type = {StrTyp}, Stroke ID = {k}): '
                          f'Soil Resistivity & Rel. Permitivity (Mod = {SSdy["flag"][5]}) = '
                          f'{SSdy["dat"][i - 1, 0]}, {SSdy["dat"][i - 1, 1]}')

                # (4) Solving Algorithm
                Circuit_Generate = Circuit_Generate() # 创建Circuit_Generate类的实例
                A_Total, R_Total, L_Total, C_Total, G_Total, Tower_Nodes_0, Tower_Brans_0= Circuit_Generate.Circuit_Generate(Tower, Span, Cable, GLB)
                
                #  Generate the source position and Soc data in global system for direct lightning
                Source_Position_Generate = Source_Position_Generate() # 创建Source_Position类的实例
                Soc_Pos, Soc_Data = Source_Position_Generate.Source_Position_Generate(Tower, Span, GLB, A_Total)

                Ins_pos = []
                Ins_on = []

                Solution = Solution()
                if GLB['nle_flag'] == 0:  # linear system
                    Vnt_Total2, Ibt_Total2 = Solution.Solution_V2(A_Total, R_Total, L_Total, C_Total, GLB, Soc_Pos, Soc_Data)
                elif GLB['nle_flag'] == 1:  # nonlinear system
                    Ins_pos, Ins_type, SA_pos, SA_type = Nle_gen(Tower, Tower_Brans_0)
                    Vnt_Total2, Ibt_Total2, R_temp, Ins_on = Solution_S2_Nonlinear(A_Total, R_Total, L_Total,
                                                                                   C_Total, G_Total, GLB, Soc_Pos,
                                                                                   Soc_Data, SA_pos, SA_type, Ins_pos,
                                                                                   Ins_type)
                else:
                    Vnt_Total2 = []
                    Ibt_Total2 = []

                Tower, Span = Mea_out(GLB, Tower, Span, Tower_Nodes_0, Tower_Brans_0, Vnt_Total2, Ibt_Total2, Ins_on,
                                      Ins_pos)  # generate outputs

                # (5) append the output of each stroke to the output of a flash
                n1, n2 = Vnt_Total2.shape
                b1, b2 = Ibt_Total2.shape
                tmpV = np.zeros((n1, n2))
                tmpI = np.zeros((b2, b2))
                tmpV[:n1, :n2] = Vnt_Total2
                tmpI[:b1, :b2] = Ibt_Total2
                output[i - 1]['Vnt'].append(tmpV)
                output[i - 1]['Ibt'].append(tmpI)

            # End of sensitivity cases loop

        for ik in range(SSdy['Nca']):
            output[ik]['FO'] = np.ones(StrNum)  # 0 = FO, 1 = Non-FO

        print("******  Completed Sensitivity Study  -- %s Model  ******\n\n" % Str[SSdy_id])
        return output, Tower, Span