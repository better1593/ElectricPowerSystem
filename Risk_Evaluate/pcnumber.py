import numpy as np

def pcnumber(Line):
    # struct获取每个变量名
    # Suppose_OHLP = np.array(Line['Suppose_OHLP'])
    SWnumber = Line['SWnumber']
    Suppose_OHLP = Line['Suppose_OHLP']
    # OHLPf = np.zeros((Suppose_OHLP.shape[0], Suppose_OHLP.shape[0] + 2), dtype=object)
    # OHLPf = np.zeros((len(Suppose_OHLP), len(Suppose_OHLP[0]) + 2), dtype=object)
    # OHLPf = np.zeros((Suppose_OHLP.shape[0], 1 + 2), dtype=object)
    OHLPf = np.zeros((len(Suppose_OHLP), 1 + 3), dtype=object)

    for pw in range(len(Suppose_OHLP)):
        Suppose_OHLPe = Suppose_OHLP[pw]
        dh_all = np.zeros((Suppose_OHLPe.shape[0], 1), dtype=object)
        dh_all[(SWnumber[0, pw] + 1):, 0] = Suppose_OHLPe[(SWnumber[0, pw] + 1):, 6]
        Indices1 = np.where(dh_all > 0)[
            0]  ## 和matlab不一样??????????????????????Indices3和Indices6不需要改变，我已经考虑了它们的变化，我需要的是位置牵引？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？
        if Indices1.size > 0:
            Indices12 = np.where(Suppose_OHLPe[Indices1, 2] == max(Suppose_OHLPe[Indices1, 2]))[0]
            Indices2 = Indices1[Indices12]
            Indices23 = np.where(Suppose_OHLPe[Indices2, 6] == max(Suppose_OHLPe[Indices2, 6]))[0]
            Indices3 = Indices2[Indices23]
        Indices4 = np.where(dh_all < 0)[0]
        if Indices4.size > 0:
            Indices45 = np.where(Suppose_OHLPe[Indices4, 2] == max(Suppose_OHLPe[Indices4, 2]))[0]
            Indices5 = Indices4[Indices45]
            Indices56 = np.where(Suppose_OHLPe[Indices5, 6] == min(Suppose_OHLPe[Indices5, 6]))[0]
            Indices6 = Indices5[Indices56]
        if 'Indices3' in locals() and 'Indices6' in locals():
            OHLPf[pw, 0] = Suppose_OHLPe[np.concatenate((np.arange(SWnumber[0, pw] + 1), Indices3, Indices6)), :]
            OHLPf[pw, 1] = [Indices3 + 1, Indices6 + 1]
            OHLPf[pw, 2] = Suppose_OHLPe[:, 7:]
            OHLPf[pw, 3] = [Indices3, Indices6]
        elif 'Indices3' in locals() and 'Indices6' not in locals():
            OHLPf[pw, 0] = Suppose_OHLPe[np.concatenate((np.arange(SWnumber[0, pw] + 1), Indices3)), :]
            OHLPf[pw, 1] = Indices3 + 1
            OHLPf[pw, 2] = Suppose_OHLPe[:, 7:]
            OHLPf[pw, 3] = [Indices3]
        elif 'Indices3' not in locals() and 'Indices6' in locals():
            OHLPf[pw, 0] = Suppose_OHLPe[np.concatenate((np.arange(SWnumber[0, pw] + 1), Indices6)), :]
            OHLPf[pw, 1] = Indices6 + 1
            OHLPf[pw, 2] = Suppose_OHLPe[:, 7:]
            OHLPf[pw, 3] = [Indices6]
        else:
            OHLPf[pw, 0] = Suppose_OHLPe[:(SWnumber[0, pw] + 1), :]
            OHLPf[pw, 1] = np.nan
            OHLPf[pw, 2] = np.nan
            OHLPf[pw, 3] = np.nan

        # 删除多个变量
        try:
            del Indices3, Indices6
        except NameError:
            pass  # 如果变量不存在，忽略错误
    # OHLPf = list(list(OHLPf))
    return OHLPf