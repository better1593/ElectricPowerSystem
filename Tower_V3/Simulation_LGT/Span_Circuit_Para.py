def Span_Circuit_Para(Info, OHLP, GND):
    VFmod = Info[7:8]  # Assuming Info is a list or tuple
    VFIT = []  # Assuming VFIT is not used in the Python version
    Para = OHL_Para_Cal(OHLP, VFmod, VFIT, GND)
    return Para