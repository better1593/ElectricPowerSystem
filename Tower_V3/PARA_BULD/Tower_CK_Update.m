function [CK_Para,Node,Bran,Meas]=Tower_CK_Update...
    (CK_Para,Node,Bran,Meas,CK_X,NodeX,BranX,MeasX,Flag)
% Update CK_Para (A,R,L,c,G,Vs,Is,Nle and Swh), Node,Bran,Meas with
%                 CK_X,NodeX,BranX,MeasX using NodeX.com/comdex
%   CK_X.C.list = [n1 n2] /listdex = [n1 n2 val]
%   CK_X.G.list = [n1 n2] /listdex = [n1 n2 val mod], mod = 1(vcis), 2 = G
%   Special issue: (1) 0 in n1 and n2 retains 0 after megering
%                  (2) duplicate common nodes should be deleted after merg.

if Flag ==0
    return;
end

% (0) Update Node/Brann.num
Ncom = size(NodeX.com,1);                           % common node #
Node.num = Node.num + NodeX.num - [Ncom Ncom 0 0];
Bran.num = Bran.num + BranX.num;

% Part A: 
% (1) update node/bran id in NodeX/BranX/MeasX (Sub-CK) 
oftn = size(Node.list,1)-Ncom;      % node_id oftn, removing common node #
oftb = size(Bran.list,1);                           % bran_id oftb
NodeX.listdex = NodeX.listdex+oftn;             
BranX.listdex(:,1) = BranX.listdex(:,1)+oftb;
BranX.listdex(:,2:3) = BranX.listdex(:,2:3)+oftn;
MeasX.listdex(:,1) = MeasX.listdex(:,1)+oftb;
MeasX.listdex(:,2:3) = MeasX.listdex(:,2:3)+oftn;
if ~isempty(CK_X.C.list)
    CK_X.C.listdex(:,1:2) = CK_X.C.listdex(:,1:2) + oftn;
    CK_X.C.listdex(:,1:2) = InfNodeUpdate(CK_X.C.listdex(:,1:2), oftn);
end
if ~isempty(CK_X.G.list)
    CK_X.C.listdex(:,1:2) = CK_X.G.listdex(:,1:2) + oftn;
    CK_X.C.listdex(:,1:2) = InfNodeUpdate(CK_X.C.listdex(:,1:2), oftn);
end

% Retain 0 for the inf. node
BranX.listdex(:,2:3) = InfNodeUpdate(BranX.listdex(:,2:3), oftn);
MeasX.listdex(:,2:3) = InfNodeUpdate(MeasX.listdex(:,2:3), oftn);

% (2) Replace common nodes of NoteX/BranX/MeasX/CK_X.C/G with GLB node 
for i = 1:Ncom
    str1 = NodeX.com(i,1);                          % Global name
    str2 = NodeX.com(i,2);                          % Local name
    num1 = NodeX.comdex(i,1);                       % Global id

    % Node updating
    tmp0 = find(NodeX.list==str2);
    NodeX.list(tmp0) = str1;
    NodeX.listdex(tmp0) = num1;
    
    %$% Node (C.list/listdex) updating
    if ~isempty(CK_X.C.list)
        tmp0 = find(CK_X.C.list==str2);
        CK_X.C.list(tmp0) = str1;
        CK_X.C.listdex(tmp0) = num1;
    end

    %$% Node (G.list/listdex) updating
    if ~isempty(CK_X.G.list)
        tmp0 = find(CK_X.G.list==str2);
        CK_X.G.list(tmp0) = str1;
        CK_X.G.listdex(tmp0) = num1;
    end

    % Bran updating
    tmp0 = find(BranX.list==str2);                  % All nodes updated
    if ~isempty(tmp0)                               % +++
        BranX.list(tmp0) = str1;
        BranX.listdex(tmp0) = num1;    
    end
    % Meas updating
    tmp0 = find(MeasX.list==str2);
    if ~isempty(tmp0)                               % +++
        MeasX.list(tmp0) = str1;
        MeasX.listdex(tmp0) = num1;
    end
end
%--------------------------------------------------------------------------

% Part B: 
% (1) Updating overall Note/Bran/Meas    
Node.list = [Node.list; NodeX.list];
Node.list = unique(Node.list,'rows','stable');
Bran.list = [Bran.list; BranX.list];
Meas.list = [Meas.list; MeasX.list];
Node.listdex = [Node.listdex; NodeX.listdex];
Node.listdex = unique(Node.listdex,'rows','stable');
Bran.listdex = [Bran.listdex; BranX.listdex];
Meas.listdex = [Meas.listdex; MeasX.listdex];


Meas.flag = [Meas.flag; MeasX.flag];
%$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

% (2) Updating Vs Is Nle and Swh
CK_Para.Vs = Lump_Souce_Update(CK_Para.Vs,CK_X.Vs,NodeX,oftn,oftb);
CK_Para.Is = Lump_Souce_Update(CK_Para.Is,CK_X.Is,NodeX,oftn,oftb);
CK_Para.Nle = Lump_Souce_Update(CK_Para.Nle,CK_X.Nle,NodeX,oftn,oftb);
CK_Para.Swh = Lump_Souce_Update(CK_Para.Swh,CK_X.Swh,NodeX,oftn,oftb);

% (3) Update Circuit Parameters
CK_Para.R = blkdiag(CK_Para.R,CK_X.R);
CK_Para.L = blkdiag(CK_Para.L,CK_X.L);

A = blkdiag(CK_Para.A,CK_X.A);
del_col = [];                                    % +++
for i = 1:Ncom
    num1 = NodeX.comdex(i,1);                    % Global id
    num2 = NodeX.comdex(i,2);                    % Local id
    A(:,num1) = A(:,num1) + A(:,oftn+Ncom+num2);
    del_col = [del_col oftn+Ncom+num2];      % +++
end
% del_col = NodeX.comdex(:,2)+oftn+Ncom;
A(:,del_col)= [];
CK_Para.A = A;

Noft = Node.num(1) - (oftn + Ncom);
Ca = zeros(Noft,Noft);                    % additional Ca
Cx = blkdiag(CK_Para.C, Ca);
CK_Para.C = AssignCValue(Cx,CK_X.C.listdex);

Ga = zeros(Noft,Noft);                    % additional Ga
Gx = blkdiag(CK_Para.G, Ga);
CK_Para.G = AssignGValue(Gx,CK_X.G.listdex);
end