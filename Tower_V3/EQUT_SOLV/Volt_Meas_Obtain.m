function OutVn = Volt_Meas_Obtain(Vnt, nodelist)
% Return node voltage (size(Nodelist,2)=1) or voltage difference (2)
% Nodelist = [ n1 n2]

% (1) Result from the first column
MeasVn = nodelist(:,1);
Izero = find(MeasVn==0);                           % V measurement
MeasVn(Izero,1) = 1;
OutVn = Vnt(MeasVn);
OutVn(Izero) = 0;

% (2) Result from the second column
if size(nodelist,2)==2
    MeasVn = nodelist(:,2);
    Izero = find(MeasVn==0);     
    MeasVn(Izero,1) = 1;
    tmp = Vnt(MeasVn);
    tmp(Izero)=0;
    OutVn = OutVn - tmp;
    
end
end