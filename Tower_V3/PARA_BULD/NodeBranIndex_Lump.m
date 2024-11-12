function [Node,Bran,Meas,nodebran]=NodeBranIndex_Lump(data,NodeCom,Nwir)
% Return Name and ID for Node and Bran, notebran of a lump CK
% Node.list/listdex/num
% Bran.list/listdex/num
% Meas.list/listdex
% With common node/bran and Nwir.nair/ngnd/nmea/comp
% notebran = [b0 n1 n2] for all 

% (1) Initialization
if isempty(NodeCom)                         % it will never happen
    Node.list = string([]);
    Node.listdex =[];
    oftn = 0;
else                                        % read from the headlines
    Node.list = NodeCom.list;
    Node.listdex =NodeCom.listdex;
    oftn =size(NodeCom.listdex,1);
end
Node.pos =[];
Bran.pos = [];

% (2) Assign node id and bran id for every line
namelist = convertCharsToStrings(data(:,3:5)); % string array of [b0 n1 n2]
Nrow = size(data,1);
% (2a) find out bran.list/listdex/num
elist = ["RL" "R" "L" "nle" "swh" "Vs"];     % single line for Bran counting
Bran.list = string([]);
oftb = 0;
for i = 1:Nrow
% (1) bran indexing 
    str = data{i,1};                    % 1st field = char
    if ~isempty(find(elist==str))       % ["R" "L" "nle" "swh" "Vs"]
         Bran.list = [Bran.list; namelist(i,1:3)];
         oftb = oftb + 1;
    end
    if str=="icvs"             % "icvs" (1+2 lins)
         Bran.list = [Bran.list; namelist(i+[0 1],1:3)];
         oftb = oftb + 2;
    end
    if str=="M2"               % "M2" (1+3 lines)
         Bran.list = [Bran.list; namelist(i+[0 2],1:3)];
         oftb = oftb + 2;
    end
    if str=="M3"               % "M3" (1+4+6 lines)
         Bran.list = [Bran.list; namelist(i+[0 3 5],1:3)];
         oftb = oftb + 3;
    end
% -------------------------------------------------------------------------
% Get matching pole impedance network (a)
    if str=="ML"
         Bran.list = [Bran.list; namelist(i+(0:oftn-1),1:3)];
         oftb = oftb + oftn;        
    end
%--------------------------------------------------------------------------
end
Bran.num = [oftb 0 0 oftb 0 0];

% (2a) find out Node.list/listdex/num and bran.list/listdex
nodebran = [];
for i = 1:Nrow
    tmp0 = zeros(1,3);                  % [b0 n1 n2]
% (1) bran indexing 
    str = string(namelist{i,1});                % bran name
    if ~strcmp(str," ") && ~strcmp(str,"") && ~ismissing(str)   
        tmp0(1) = find(Bran.list(:,1)==str);
    end
          
% (2) node indexing 
    for j=1:2                           % for one node of two
        str = string(namelist{i,1+j});  % node name in con. 4  & 5
        if ~strcmp(str," ") && ~strcmp(str,"") && ~ismissing(str)
            tmp1 = find(Node.list==str);% string array                
            if isempty(tmp1)
                oftn = oftn+1;
                Node.list = [Node.list; str];
                Node.listdex = [Node.listdex; oftn];
                tmp0(1+j) = oftn;
            else
                tmp0(1+j) = tmp1;
            end
        end
    end 
    nodebran = [nodebran; tmp0];        % [b0 n1 n2] for all lines
end
Node.num = [oftn oftn 0 0]; 

Bran.listdex = zeros(oftb,3);
Bran.listdex(:,1) = (1:oftb)';
Bran = AssignElemID(Node,Bran,[2 3]);

% (3) Measurement list
% Ncop = Nwir.comp;
% Meas.list = namelist(Ncop+1:end,1:3);
% Meas.listdex = nodebran(Ncop+1:end,1:3);
tmp = [data{:,2}]';                             % meas flag
Itmp = find(tmp>0);                             % find id of numeric items
Meas.list = namelist(Itmp,:);
Meas.listdex = nodebran(Itmp,1:3);
Meas.flag = tmp(Itmp);
% Meas = AssignElemID(Bran,Meas,1);               % !!! not ncessary !!!
%$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

end






