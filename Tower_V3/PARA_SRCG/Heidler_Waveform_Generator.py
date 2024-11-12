import numpy as np

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