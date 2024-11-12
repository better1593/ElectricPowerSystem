function [DATA,Blok,Info,Nwir,GND]=GeneInfoRead_V2(data)
% [num,txt,raw] = xlsread(filename) 
% Return (1) DATA with main body of the Wire/Lump elements
%        (2) Blok.*** contains headline/ common node info. 
%            Block.flag/name of sub CK (Tower); Cir.num/dat (Span/Cable)
%                 .A2GB.list (name of A2G pairs)
%                 .sub_CK: com_node name of sub-CK for Tower (tower name)
%                          com_node name of sun-CK for sub-CK (local name)
%        (2) Info (T/S/C), GND, END (sub-CK name as the suffix of node/bran
%        (3) Nwir.nair/ngnd/nmea/comp   
%        (4) Cir.dat/num
%            Seg.num = [tot# pha# sw# slen seg# depth]
%        (4) Read TOWEER. CABLE, SPAN, GND, INFO, MEAS, END, INSU, SARR,
%            TXFM,A2GB,GRID,INVT,INTF,MTCK,OTH1,OTH2,AirW,GndW,POLE,
%        Note: "" '0' means this is an empty node (to be deleted)
    
% (1) Remove comment lines starting with %
Nrow = size(data,1);                            % cell stru
row_del = [];
for i = 1:Nrow
    str = data{i,1};
    if str(1)=='%'
        row_del = [row_del i];
    end
end
DATA = data;                                    % cell array
DATA(row_del,:) = [];

BTYPE = DATA{1, 1};                             % Block Type (TOWER/LUMP)
COL = 0;                                        % col for sub_CK file name
if strcmp(BTYPE,'TOWER')
    COL = 24;   
end

% (2) Read  general info. about T/S/C and L about com_node (name and/or id)
Info = [];
GND = [];
Cir = [];
blokflag = zeros(1,7);
blokname = [];
bloktow = [];
blokins = [];
bloksar = [];
bloktxf = [];
blokgrd = [];
blokint = [];   % + inveter (simple)
blokinf = [];   % + inveter (full)
blokmck = [];   % + matching circuit
blokoth1 = [];  % + other 1
blokoth2 = [];  % + other 2
bloka2g = [];                                   % air-2-gnd bridge
bloklup = [];
sys_name = [];

nair = 0;
ngnd = 0;
nmea = 0;
row_del = [];
nrow = size(DATA,1);               % # of lines with [b0, n1 n2] names
for i = 1:nrow                                  % read data_para table   
    firstColumnValue = string(DATA{i, 1});      % check the key word
    switch firstColumnValue
        case "TOWER"
            row_del = [row_del i];
            blokflag(1:10) = cell2mat(DATA(i,2:11)); % + T: sub_CK flag 
            blokname = ["WIRE" "" "" "" "" "" "" "" "" ""]; % + 
        case "SPAN"
            row_del = [row_del i:i+1];
            Cir.num = cell2mat(DATA(i:i+1,2:7));% S: Cir.dat (2 lines)
        case "CABLE"
            row_del = [row_del i:i+1];
            Cir.num = cell2mat(DATA(i:i+1,2:7));% C: Cir.dat (2 lines)
        case "INFO"     
            row_del = [row_del i];
            Info = DATA(i,2:13);                % T/S/C Info (cell array)
        case "GND"
            row_del = [row_del i];              % T/S/C soil model
            GND.glb = DATA{i,2};                % 1 = GLB_GND_data
            GND.sig = DATA{i,6};
            GND.mur = DATA{i,7};
            GND.epr = DATA{i,8};
            GND.gnd = DATA{i,9};                % gnd model: 0 or 1 or 2
        case "A2GB"                             % T: air-2-gnd conn pair
            blokflag(1) = 2;            % + 1 = AirW only, 2 = AirW+GndW
            blokname(1) = "Wire+A2G";   % +
            bloka2g.list = string(zeros(0,3));
            rownum = DATA{i, 6};                % line # for input vari
            oft = 0;
            for j = 1:2:rownum
                rowid = i+(j-1);
                for k=2:5
                    tmp1=string(DATA{rowid,k});         %  Air node name
                    tmp2=string(DATA{rowid+1,k});       %  Gnd node name
                    if ~ismissing(tmp1)& ~strcmp(tmp1," ")
                        oft =oft + 1;
                        tmp0 = "S" + num2str(oft,'%02d');
                        bloka2g.list = [bloka2g.list; [tmp0 tmp1 tmp2]];
                    end
                end
            end
            bloka2g.listdex = [];
            row_del = [row_del i+(0:rownum-1)];
        case "INSU"
            [row_ins,blokins,blokname(2)] = ComNodeRead(DATA,i,COL); 
            row_del = [row_del row_ins];
        case "SARR"
            [row_sar,bloksar,blokname(3)] = ComNodeRead(DATA,i,COL);            
            row_del = [row_del row_sar];
        case "TXFM"
            [row_txf,bloktxf,blokname(4)] = ComNodeRead(DATA,i,COL);            
            row_del = [row_del row_txf];
        case "GRID"
            [row_grd,blokgrd,blokname(5)] = ComNodeRead(DATA,i,COL);            
            row_del = [row_del row_grd];
        case "INVT"
            [row_int,blokint,blokname(6)] = ComNodeRead(DATA,i,COL);            
            row_del = [row_del row_int];
        case "INVF"
            [row_inf,blokinf,blokname(7)] = ComNodeRead(DATA,i,COL);            
            row_del = [row_del row_inf];
        case "MTCK"
            [row_mck,blokmck,blokname(8)] = ComNodeRead(DATA,i,COL);            
            row_del = [row_del row_mck];
        case "OTH1"
            [row_oth1,blokoth1,blokname(9)] = ComNodeRead(DATA,i,COL);            
            row_del = [row_del row_oth1];
        case "OTH2"
            [row_oth2,blokoth2,blokname(10)] = ComNodeRead(DATA,i,COL);            
            row_del = [row_del row_oth2];
       case "END"                           % Used for loading lump model
            row_del = [row_del i];
            sys_name = string(DATA{i,2});   % Lump sub-CK name
        case "AirW"
            nair = nair + 1;                % record air wire #
        case "GndW"
            ngnd = ngnd + 1;                % record gnd wire #
        case "Meas"                         % record the meas line #
            nmea = DATA{i,6};               % 
        case "LUMP"                         % read com_node in lump CK file
            [row_lup,bloklup,notused] = ComNodeRead(DATA,i,COL);  
            row_del = [row_del row_lup];    % 
    end
end
Blok.sysname = sys_name;
Blok.Cir = Cir;
Seg = [];
if ~isempty(Cir)
    Seg.Ncon = Cir.num(2,1);
    Seg.Npha = Cir.num(2,1)-Cir.num(2,2);
    Seg.Ngdw = Cir.num(2,2);
    Seg.num=[Seg.Ncon Seg.Npha Seg.Ngdw]; 
end
Blok.Seg = Seg;
Blok.flag = blokflag;
Blok.name = blokname;

Blok.tow = bloktow;
Blok.ins = blokins;
Blok.sar = bloksar;
Blok.txf = bloktxf;
Blok.grd = blokgrd;
Blok.int = blokint;
Blok.inf = blokinf;
Blok.mck = blokmck;
Blok.oth1 = blokoth1;
Blok.oth2 = blokoth2;
Blok.a2g = bloka2g;                 % Air-2-Gnd Bridge (agb)
Blok.lup = bloklup;                 % read com_node in sub-CK file

DATA(row_del,:) = [];
nrow = size(DATA,1);               % # of lines with [b0, n1 n2] names\
Nwir.nair = nair;
Nwir.ngnd = ngnd;
Nwir.nmea = nmea;
Nwir.comp = nrow - nmea;            % # of all wires or lump components
end