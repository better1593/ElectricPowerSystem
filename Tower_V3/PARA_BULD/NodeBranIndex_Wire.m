function [Node,Bran,Meas,Blok,nodebran]=NodeBranIndex_Wire(data,Blok,Nwir)
% Return (1) Name and ID for Node, Bran and Meas of Wires and 
%        (2) Updated id of Common Nodes in Blok
%       Node.list/listdex/num/pos
%       Bran.list/listdex/num
%       Meas.list/listdex
% With data (cell array), common node/bran, Nwir.nair/ngnd/nmea/comp

% (0) Initialization
flag = Blok.flag;               % +

Node.list = string([]);
Node.listdex =[];
Node.pos = [];                  % cooridnates of nodes
Node.com =[];
Node.comdex = [];
Node.num = zeros(1,4);
Node.pos = [];

Bran.list = [];
Bran.listdex = [];
Bran.list = [];
Bran.pos = [];
Bran.num = zeros(1,6);

Meas.list = [];
Meas.listdex = [];
Meas.flag = [];

oftn = 0;
Blist = string([]);
Blistdex =[];
nodebran = [];

Nrow = size(data,1);
Ncop = Nwir.comp;               % # of componenets
Nmea = Nwir.nmea;               % # of measurment lines
Nair = Nwir.nair;               % air wire #
Ngnd = Nwir.ngnd;               % gnd wire #


if Nrow ~=0
% (1) bran indexing 
Nnum = zeros(1,4);
namelist = convertCharsToStrings(data(:,3:5)); % string array [b0 n1 n2]
for i = 1:Nrow
    tmp0 = zeros(1,3);                      % [b0 n1 n2]
    tmp0(1) = i;
% (2) node indexing 
    for j=1:2                               % for one node of two
        str = namelist(i,1+j);  % node name in con. 4  & 5
        if ~strcmp(str,"") & ~strcmp(str," ") & ~ismissing(str)

            tmp1 = find(Node.list==str);    % string array                
            if isempty(tmp1)
                oftn = oftn+1;
                Node.list = [Node.list; str];
                Node.listdex = [Node.listdex; oftn];
                pos =  cell2mat(data(i,(j-1)*3+(6:8)));
                Node.pos = [Node.pos; pos];
                tmp0(1+j) = oftn;
            else
                tmp0(1+j) = tmp1;               
            end
        end
    end 
    Blistdex = [Blistdex; tmp0];
    nodebran = [nodebran; tmp0];    % [b0 n1 n2]
    
    if i == Nair                    % for wire case
        Nnum(2) = oftn;
        Nnum(4) = oftn;
    end
    if i == Nair + Ngnd             % for wire case
        Nnum(3) = oftn - Nnum(2);
    end
end
Nnum(1) = oftn;
Node.num = Nnum; 

Bran.list = namelist(1:Ncop,1:3);
Bran.listdex = nodebran(1:Ncop,1:3);
Bran.num = [Ncop Nair Ngnd 0 0 0];              % wire parameters

% Meas.list = namelist(Ncop+1:end,1:3);
% Meas.listdex = nodebran(Ncop+1:end,1:3);
tmp = [data{:,2}]';                             % meas flag
Itmp = find(tmp>0);                             % find id of numeric items
Meas.list = namelist(Itmp,:);
Meas.listdex = nodebran(Itmp,1:3);
Meas.flag = tmp(Itmp);
Meas = AssignElemID(Bran,Meas,1);               % !!! not ncessary !!!
%$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

else % without connecting wire in the tower (LUMP only)
    oft = 0;
    [Node, oft] = AssignElemLump(Node,Blok.ins,oft);      
    [Node, oft] = AssignElemLump(Node,Blok.sar,oft);      
    [Node, oft] = AssignElemLump(Node,Blok.txf,oft);      
    [Node, oft] = AssignElemLump(Node,Blok.grd,oft);      
    [Node, oft] = AssignElemLump(Node,Blok.int,oft);      
    [Node, oft] = AssignElemLump(Node,Blok.inf,oft);      
    [Node, oft] = AssignElemLump(Node,Blok.mck,oft);      
    [Node, oft] = AssignElemLump(Node,Blok.oth1,oft);      
    [Node, oft] = AssignElemLump(Node,Blok.oth2,oft);      
end

% (3)  update blok.sys.listdex (Com_Node id in WIRE for all sub-cir)
if flag(2)==1
    Blok.ins = AssignElemID(Node,Blok.ins,1);
end
if flag(3)==1
    Blok.sar = AssignElemID(Node,Blok.sar,1);
end
if flag(4)==1
    Blok.txf = AssignElemID(Node,Blok.txf,1);
end
if flag(5)==1
    Blok.grd = AssignElemID(Node,Blok.grd,1);
end
if flag(6)==1
    Blok.int = AssignElemID(Node,Blok.int,1);
end
if flag(7)==1
    Blok.inf = AssignElemID(Node,Blok.inf,1);
end
if flag(8)==1
    Blok.mck = AssignElemID(Node,Blok.mck,1);
end
if flag(9)==1
    Blok.oth1 = AssignElemID(Node,Blok.oth1,1);
end
if flag(10)==1
    Blok.oth2 = AssignElemID(Node,Blok.oth2,1);
end
end






