function [Cable C2Tmap]= Cable_Circuit_Build...
          (CableModelName,TCom,TH_Info,TH_Node,TT_Info,TT_Node,VFIT,GLB)
% TCom.list/listdex [T_head T_tail C_head C_tail]
% Seg.Nseg/Ncon/Nphas/Ngdw/Lseg
% Seg.num = [tot# pha# sw# slen seg# depth]

% (1) 初始化变量，Read a complete table                                    
% data0 = readtable(CableModelName);      
[num,txt,raw_data] = xlsread(CableModelName);
%--------------------------------------------------------------------------

% (2) Read general info. of a Cable
[data,Blok,Info,Nwir,GND] = GeneInfoRead_V2(raw_data);
Cable.Info = Info;
Cable.ID = Info{1,10};
Cable.Atn = GLB.Acab(Cable.ID,:);
if GND.glb==1
   Cable.GND = GLB.GND;
end
% modified------------------------------------------------------------------------
Blok.Seg.Lseg = GLB.slg;                       % length of a segment
Blok.Seg.Npha = Info{1,5};
Blok.Seg.Ngdw = Info{1,6};                       
Blok.Seg.Ncon = Blok.Seg.Npha + Blok.Seg.Ngdw;
Blok.Seg.num = [Blok.Seg.Ncon Blok.Seg.Npha Blok.Seg.Ngdw Blok.Seg.Lseg]; 
% ------------------------------------------------------------------------

% (2b) Read relative info. of T_head, T_tail
Cable.Info(1,3) = TH_Info(1,1);                 % name of head tower
Cable.Info(1,4) = TT_Info(1,1);                 % name of tail tower
Pole(1,1:3) = [TH_Info{1,5:7}];                 % pos of head tower
Pole(1,4:6) = [TT_Info{1,5:7}];                 % pos of tead tower
% Cable.Info(1,5) = TH_Info(1,10);                % id of head tower
% Cable.Info(1,6) = TT_Info(1,10);                % id of tead tower
rng = 1;
TC_head = TCom.head;
TC_tail = TCom.tail;
TC_head= AssignElemID(TH_Node,TC_head,rng);     % Tnode ID and local pos
TC_tail= AssignElemID(TT_Node,TC_tail,rng);     % Tnode ID and local pos
%--------------------------------------------------------------------------

% (3) Cable Parameters
% (3a) Initilization of Cable parameters (Order: SWA + Core A/B/C/N)
[Node,Bran,Meas,Cir,Seg]=NodeBranIndex_Line(data,Blok,Pole);
Node.com = [Node.com(:,1) TC_head.list Node.com(:,2) TC_tail.list];
Node.comdex = [Node.comdex(:,1) TC_head.listdex ...
               Node.comdex(:,2) TC_tail.listdex]; 
Ncom = size(Node.com,1);                    % # of common nodes
for i = 1:Ncom
    Node.pos(i,:) = [TC_head.pos(i,:) TC_tail.pos(i,:)] + Pole;
end

% (3b) Read wire parameters of a cable
Ncab = Cir.num(1,6);                        % # of line/conductors
Seg.Ncab = Ncab;
OHLP = data(1:Ncab,6:20);                   % Cell: Cable data only

% Getting cooridnates for Pole and lines
Line.pos = Pole;                            % Pole start/end positions
Line.rad = [OHLP{1,1:6}];                   % wire radius and height
Line.mat = [OHLP{1,9:13}];            % wire mat: sigc/siga/murc/mura/epri
Line.con = Seg.num;                         % total# core# arm# seg#

% (3d) Mapping table (hspan/hsid/Thead tspan/tsid/Ttail)
C2Tmap.head = [[Cable.ID TH_Info{1,10}]; Node.comdex(:,1:2)];
C2Tmap.tail = [[Cable.ID TT_Info{1,10}]; Node.comdex(:,3:4)];

%--------------------------------------------------------------------------

% (4) Calculate wave equation parameters (Z and Y)
Para = Cable_Para_Cal(Line,VFIT,GND);

% (5) Output---------------------------------------------------------------
Cable.Cir = Cir;
Cable.Seg = Seg;
Cable.Pole = Pole;
Cable.Line = Line;
Cable.Para = Para;
Cable.Node = Node;
Cable.Bran = Bran;
Cable.Meas = Meas;
Cable.C2Tmap = C2Tmap; 
Cable.Soc = [];
end



