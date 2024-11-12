import numpy as np
from Tower_V3.PARA_SRCG.Heidler_Waveform_Generator import Heidler_Waveform_Generator
from Tower_V3.PARA_SRCG.CIGRE_Waveform_Generator import CIGRE_Waveform_Generator

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