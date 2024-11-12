function [CK_Para,Node,Bran,Meas]=Lump_Model_Intepret(Flag,Name,ID,TCOM)
% Build Model of a lump sub-system with input from an excel file
% Input: Flag = 1/0 enable/disable 
%        Name = input file name
%        ID =   sub-system ID (2=INS, 3=SAR, 4=TXF, 5=GRD, 6=OTH)
%        BCOM = Nodecom.list/listdex (external/local)

% (1) 初始化变量，用于存储提取的数据
if Flag(ID)==0
    CK_Para = []; Bran = []; Node =[]; Meas = [];
    return;
end
LumpModelName = Name(ID);
% data0 = readtable(LumpModelName);     % read a complete table
[num,txt,raw_data] = xlsread(LumpModelName);
%--------------------------------------------------------------------------

% (2a) Read common node (node.com/comdex/suffix=sysname)
[data,Blok,Info,Nwir,notused] = GeneInfoRead_V2(raw_data);
NodeCom_int = Blok.lup;                             % com_node in sub_CK
Ncom = length(Blok.lup.list);                       % + node # for mck only

% (2b) Read bran/node/meas/num, nodebran and assign them with ID
[Node,Bran,Meas,nodebran] = NodeBranIndex_Lump(data,NodeCom_int,Nwir);

% (2c) Update node.com/comdex including both int anx ext info.
Node.com = [TCOM.list NodeCom_int.list];           % [ext int]: Node.com
Node.comdex = [TCOM.listdex NodeCom_int.listdex];  % [ext int]: Node.comdex

% (2d) Add the suffix into the name of nodes/brans
app = "_" + Blok.sysname;                           % suffix
Node.list = AssignSuffix(Node.list,app);
Node.com(:,2) = AssignSuffix(Node.com(:,2),app);
Bran.list = AssignSuffix(Bran.list,app);
Meas.list = AssignSuffix(Meas.list,app);
%--------------------------------------------------------------------------

% (3) Build circuit models by groups
Nn = Node.num(1);                               % node #
Nb = Bran.num(1);                               % bran #
R = zeros(Nb,Nb);
L = zeros(Nb,Nb);
C.list = string([]);
C.listdex = [];
G.list = string([]);
G.listdex = [];
A = zeros(Nb,Nn);
Vs.dat = [];
Vs.pos = [];                            % [bran id node_1 node_2]
Is.dat = [];
Is.pos = [];                            % node id
Nle.dat = [];                           % model id
Nle.pos = [];                           % [bran id node_1 node_2]
Swh.dat = [];                           % model id
Swh.pos = [];                           % [bran id node_1 node_2]

Nline = Nwir.comp;                      % # of components
for i = 1:Nline                         % 循环遍历数据表    
    firstColumnValue = string(data{i, 1});      % 提取第一列的值
    switch firstColumnValue{1}          % 根据不同的标签，提取数据到相应的变量
        case 'RL'                        % 提取本行的数据（第3、4、5列）到R0
            A = AssignAValue(A,nodebran(i,1),nodebran(i,2),-1);
            A = AssignAValue(A,nodebran(i,1),nodebran(i,3),+1);
            R(nodebran(i,1),nodebran(i,1)) = data{i,6};       
            L(nodebran(i,1),nodebran(i,1)) = data{i,7};       
        case 'R'                        % 提取本行的数据（第3、4、5列）到R0
            A = AssignAValue(A,nodebran(i,1),nodebran(i,2),-1);
            A = AssignAValue(A,nodebran(i,1),nodebran(i,3),+1);
            R(nodebran(i,1),nodebran(i,1)) = data{i,6};       
        case 'L'                        % 提取本行的数据（第3、4、5列）到L0
            A = AssignAValue(A,nodebran(i,1),nodebran(i,2),-1);
            A = AssignAValue(A,nodebran(i,1),nodebran(i,3),+1);
            L(nodebran(i,1),nodebran(i,1)) = data{i,6};       
        case 'C'                        % 提取本行的数据（第3、4、5列）到C0
            C.list = [C.list; convertCharsToStrings(data(i,4:5))];% Node name
            C.listdex=[C.listdex; [nodebran(i,2),nodebran(i,3),data{i,6}]];
        case 'G'                        % 提取本行的数据（第3、4、5列）到G0
            G.list = [G.list; convertCharsToStrings(data(i,4:5))];% Node name
            G.listdex=[G.listdex; [nodebran(i,2),nodebran(i,3),data{i,6}],2];
        case 'M2'               % RL Bran, R: data(*,6) + L:data(*,7)2
            A = AssignAValue(A,nodebran(i,1),nodebran(i,2),-1);
            A = AssignAValue(A,nodebran(i,1),nodebran(i,3),+1);
            A = AssignAValue(A,nodebran(i+2,1),nodebran(i+2,2),-1);
            A = AssignAValue(A,nodebran(i+2,1),nodebran(i+2,3),+1);
            R(nodebran(i,1),nodebran(i,1)) = data{i,6};       
            R(nodebran(i,1),nodebran(i+2,1)) = data{i+1,6};       
            R(nodebran(i+2,1),nodebran(i,1)) = data{i+1,6};       
            R(nodebran(i+2,1),nodebran(i+2,1)) = data{i+2,6};       
            L(nodebran(i,1),nodebran(i,1)) = data{i,7};       
            L(nodebran(i,1),nodebran(i+2,1)) = data{i+1,7};       
            L(nodebran(i+2,1),nodebran(i,1)) = data{i+1,7};       
            L(nodebran(i+2,1),nodebran(i+2,1)) = data{i+2,7};       
       case 'M3'               % RL Bran, R: data(*,6) + L:data(*,7)
            A = AssignAValue(A,nodebran(i,1),nodebran(i,2),-1);
            A = AssignAValue(A,nodebran(i,1),nodebran(i,3),+1);
            A = AssignAValue(A,nodebran(i+3,1),nodebran(i+3,2),-1);
            A = AssignAValue(A,nodebran(i+3,1),nodebran(i+3,3),+1);
            A = AssignAValue(A,nodebran(i+5,1),nodebran(i+5,2),-1);
            A = AssignAValue(A,nodebran(i+5,1),nodebran(i+5,3),+1);

            R(nodebran(i,1),nodebran(i,1)) = data{i,6};       
            R(nodebran(i+3,1),nodebran(i+3,1)) = data{i+3,6};       
            R(nodebran(i+5,1),nodebran(i+5,1)) = data{i+5,6};  
                         
            R(nodebran(i,1),nodebran(i+3,1)) = data{i+1,6};       
            R(nodebran(i+3,1),nodebran(i,1)) = data{i+1,6};       
            R(nodebran(i,1),nodebran(i+5,1)) = data{i+2,6};       
            R(nodebran(i+5,1),nodebran(i,1)) = data{i+2,6};       
            R(nodebran(i+3,1),nodebran(i+5,1)) = data{i+4,6};       
            R(nodebran(i+5,1),nodebran(i+3,1)) = data{i+4,6};              
            
            L(nodebran(i,1),nodebran(i,1)) = data{i,7};       
            L(nodebran(i+3,1),nodebran(i+3,1)) = data{i+3,7};       
            L(nodebran(i+5,1),nodebran(i+5,1)) = data{i+5,7};  
             
            L(nodebran(i,1),nodebran(i+3,1)) = data{i+1,7};       
            L(nodebran(i+3,1),nodebran(i,1)) = data{i+1,7};       
            L(nodebran(i,1),nodebran(i+5,1)) = data{i+2,7};       
            L(nodebran(i+5,1),nodebran(i,1)) = data{i+2,7};       
            L(nodebran(i+3,1),nodebran(i+5,1)) = data{i+4,7};       
            L(nodebran(i+5,1),nodebran(i+3,1)) = data{i+4,7};       
        case 'nle'              % 提取本行的数据（第3、4、5列）到R0
            A = AssignAValue(A,nodebran(i,1),nodebran(i,2),-1);
            A = AssignAValue(A,nodebran(i,1),nodebran(i,3),+1);
            R(nodebran(i,1),nodebran(i,1)) = data{i,6}; 
            Nle.pos = [Nle.pos; nodebran(i,1:3)];
            Nle.dat = [Nle.dat; string(data{i,9})];
        case 'swh'              % 提取本行的数据（第3、4、5列）到R0
            A = AssignAValue(A,nodebran(i,1),nodebran(i,2),-1);
            A = AssignAValue(A,nodebran(i,1),nodebran(i,3),+1);
            R(nodebran(i,1),nodebran(i,1)) = data{i,6}; 
            Swh.pos = [Swh.pos; nodebran(i,1:3)];
            Swh.dat = [Swh.dat; string(data{i,9})];
        case 'Is'               % 提取下一行的数据（第3、4、5列）到R0
            tmp2 = string(data{i,9});   % file name
            if ~strcmp(tmp2,"")
                Is.dat = [Is.dat {readstruct(tmp2)}];  % Cell array
            else
                ispeak = data{i,7};
                isfreq = data{i,8};
                Is.dat = [Is.dat {NaN isfreq ispeak}];  % Cell array
            end
            Is.pos = [Is.pos; nodebran(i,2)];           % Node ID
        case 'Vs'               % 提取下一行的数据（第3、4、5列）到R0    
            A = AssignAValue(A,nodebran(i,1),nodebran(i,2),-1);
            A = AssignAValue(A,nodebran(i,1),nodebran(i,3),+1);
            R(nodebran(i,1),nodebran(i,1)) = data{i,6};  
            
            tmp2 = string(data{i,9});   % file name
            if ~strcmp(tmp2,"")
                fileID = fopen(tmp2,'r');
                tmp3 = textscan(fileID,'%f');
                Vs.dat = [Vs.dat tmp3];   % Cell array
            else
                vspeak = data{i,7};
                vsfreq = data{i,8};
                Vs.dat = [Vs.dat {NaN vsfreq vspeak}]; % Cell array
            end
            Vs.pos = [Vs.pos; nodebran(i,1:3)];
        case 'vcis'             % 提取本行以及下一行的数据（第3、4、5、6列）到vcis
            G.list = [G.list; convertCharsToStrings(data(i,4:5))];% Node name
            G.listdex=[G.listdex;[nodebran(i,2),nodebran(i,3),data{i,6},1]];
%             G(nodebran(i,2),nodebran(i,3)) = data{i,6};         
        case 'icvs'             % 提取本行以及下一行的数据（第3、4、5、6列）到icvs
            A = AssignAValue(A,nodebran(i,1),nodebran(i,2),-1);
            A = AssignAValue(A,nodebran(i,1),nodebran(i,3),+1);
            A = AssignAValue(A,nodebran(i+1,1),nodebran(i+1,2),-1);
            A = AssignAValue(A,nodebran(i+1,1),nodebran(i+1,3),+1);
            
            firstColumnValue = data{i+1, 1};            % 提取第一列的值
            switch firstColumnValue{1}      % 根据不同的标签，提取数据到相应的变量
                case 'RR'
                    R(nodebran(i,1),nodebran(i+1,1)) = data{i,6};  
                    R(nodebran(i+1,1),nodebran(i+1,1)) = data{i+1,6};  
                case 'RL'
                    R(nodebran(i,1),nodebran(i+1,1)) = data{i,6};  
                    L(nodebran(i+1,1),nodebran(i+1,1)) = data{i+1,6};  
                case 'LR'
                    L(nodebran(i,1),nodebran(i+1,1)) = data{i,6};  
                    R(nodebran(i+1,1),nodebran(i+1,1)) = data{i+1,6};  
                case 'LL'
                    L(nodebran(i,1),nodebran(i+1,1)) = data{i,6};  
                    L(nodebran(i+1,1),nodebran(i+1,1)) = data{i+1,6};  
            end
% -------------------------------------------------------------------------
% Get matching pole impedance network (a)
        case 'ML'
            Vair = 3e8;                       % Velocity in free space
            Dist = [data{i+(0:Ncom-1),6}]';   % (x) without meas and others
            High = [data{i+(0:Ncom-1),7}]';   % (z) without meas and others
            r0 = mean([data{i+(0:Ncom-1),8}]);% (r) without meas and others
            [L0,C0]=Cal_LC_OHL(High,Dist,r0);
            R = L0*Vair;                      % surge impedance
            A = eye(Ncom);
    end
% -------------------------------------------------------------------------
end
CK_Para.Info = Info;
CK_Para.app = Blok.sysname;
CK_Para.A = A;
CK_Para.R = R;
CK_Para.L = L;
C.list = AssignSuffix(C.list,app);
CK_Para.C = C;                                      % C.list/lisdex
G.list = AssignSuffix(G.list,app);
CK_Para.G = G;
CK_Para.Vs = Vs;
CK_Para.Is = Is;
CK_Para.Nle = Nle;
CK_Para.Swh = Swh;
end
