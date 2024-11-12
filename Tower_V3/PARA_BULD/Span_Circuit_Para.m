function Para = Span_Circuit_Para(Info,OHLP,GND)

VFmod = [Info{1,8:9}];  
VFIT = []; % 
Para = OHL_Para_Cal(OHLP,VFmod,VFIT,GND);
end