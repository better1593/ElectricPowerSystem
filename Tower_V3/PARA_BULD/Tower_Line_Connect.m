function [Cw, Cwc] = Tower_Line_Connect(Tower, Span, Cable)
% Build a model of connecting line (Span/Cable) to a Tower using meas. bran
%       meas bran = Line node -> Tower node
%       Cwx.id =[S/C, Span_id, CK_id, in/out, tower_nid], 
%       Cwx.C0;
%       with T2S or T2C mapping table (X=[]: span, X=C: cable)

% Part A: Span only
T2Smap = Tower.T2Smap;
Cw.id = [];
Cw.C0 = [];

% (A1) Obtain Cw for OHL
hspn = T2Smap.hspn;
hsid = T2Smap.hsid;
tspn = T2Smap.tspn;
tsid = T2Smap.tsid;
for jk = 1:tspn
    map = T2Smap.tail{1,jk};
    map(1,:) = [];
    nr = size(map,1);
    Itmp = ones(nr,1);
    tmp = [Itmp Itmp*tsid(jk) Itmp*0 Itmp map(:,2)];
    Cw.id = [Cw.id; tmp];
        
    sid = tsid(jk);
    Line = Span(sid).OHLP;
    High = Line(:,6);                       % height
    Dist = Line(:,7);                       % Horizontal offset
    r0 = Line(:,8);                         % conductor radius
    [L,C] = Cal_LC_OHL(High,Dist,r0);
    Cw.C0 = [Cw.C0; diag(C)];             % diagnal elements only
end
    
for jk = 1:hspn
    map = T2Smap.head{1,jk};
    map(1,:) = [];
    nr = size(map,1);
    Itmp = ones(nr,1);
    tmp = [Itmp Itmp*hsid(jk) Itmp*0 -Itmp map(:,2)];
    Cw.id = [Cw.id; tmp];
        
    sid = hsid(jk);
    Line = Span(sid).OHLP;
    High = Line(:,3);                       % height
    Dist = Line(:,7);                       % Horizontal offset
    r0 = Line(:,8);                         % conductor radius
    [L,C] = Cal_LC_OHL(High,Dist,r0);
    Cw.C0 = [Cw.C0; diag(C)];             % diagnal elements only
end

% Part B: CABLE only
T2Cmap = Tower.T2Cmap;
Cwc.C0 = [];
Cwc.id = [];

% (B2) Obtain Cw for CABLE
hspn = T2Cmap.hspn;
hsid = T2Cmap.hsid;
tspn = T2Cmap.tspn;
tsid = T2Cmap.tsid;

for jk = 1:tspn    
    map = T2Cmap.tail{1,jk};
    map(1,:) = [];
    nr = size(map,1);
    Itmp = ones(nr,1);
    tmp = [Itmp Itmp*tsid(jk) Itmp*0 Itmp map(:,2)];
    Cwc.id = [Cwc.id; tmp];
    Para = Cable(tsid).Para;
    Cwc.C0 = [Cwc.C0; Para.Cw.C0]; 
end

for jk = 1:hspn    
    map = T2Cmap.head{1,jk};
    map(1,:) = [];
    nr = size(map,1);
    Itmp = ones(nr,1);
    tmp = [Itmp Itmp*hsid(jk) Itmp*0 -Itmp map(:,2)];
    Cwc.id = [Cwc.id; tmp];
    Para = Cable(hsid).Para;
    Cwc.C0 = [Cwc.C0; Para.Cw.C0]; 
end

end
