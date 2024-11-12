function Tower=Tower_Circuit_Build(TowerModelName,VFIT,GLB)
% Blok.flag = blokflag (lump subsystem flag list)
% Blok.name = blokname (lump subsystem name list)
% Blok.tow/ins/sar/txf/grd/oth/a2g/ (common node name with lump subsystem;
%%%% Blok.lup (lum = LUMP file name for read lump circuit model only) %%%%

% (1) Read a complete table  
Cmin = 1e-12;                    % + Min C to avoid singularity of C matrix
[num,txt,raw_data] = xlsread(TowerModelName);
%--------------------------------------------------------------------------

% (2) Read general info. of tower inc. Blok (file name, com_node name)                                      % common/gnd node and data
[data,Blok,Info,Nwir,GND]=GeneInfoRead_V2(raw_data);
Tower.Info = Info;
Tower.ID = Info{1,10};
if GND.glb==1
   Tower.GND = GLB.GND;
end
Tower.Ats = GLB.A(:,Tower.ID);
% +++++--------------------------------------------------------------------
if ~isempty(GLB.Acab) 
    Tower.Acabts = GLB.Acab(:,Tower.ID);
else
    Tower.Acabts = [];
end
%--------------------------------------------------------------------------

% (3) Build Tower Model
% (3a) Wire: Node/Bran/Meas (list/listdex/num) and Blok: sys (listdex)
[Node,Bran,Meas,Blok,nodebran]=NodeBranIndex_Wire(data,Blok,Nwir);

if Blok.flag(1)>=1                          % with wires (air or air+gnd)
%--------------------------------------------------------------------------
% (3b) Read wire parameters of a tower
    WireP = cell2mat(data(:,6:20));
    WireP=[WireP nodebran];                     % WireP(*,1:19)
    dmea = Nwir.comp + 1;
    WireP(dmea:end,:) = [];                     % delete measurement lines
% Rotate the pole according to the angle in Info(4)
    theta = Info{1,4};
    WireP(:,1:2)=RotPos(WireP(:,1:2),theta);    % update the cooridnates
    WireP(:,4:5)=RotPos(WireP(:,4:5),theta);    % update the cooridnates
    Node.pos(:,1:2) = RotPos(Node.pos(:,1:2),theta);%update the cooridnates

    Wire_Plot(WireP,Node,Bran);
    close all

% (3b) Cal. wire-Model-Parameters
    VFmod = [Info{1,8:9}];
    [CK_Para] = Wire_Model_Para3(WireP,Node,Bran,VFmod,VFIT,GND);
%--------------------------------------------------------------------------
else
    Nn = Node.num(1);
    WireP=[];    
    CK_Para.A = zeros(1,Nn);    % Add pseudo bran to retain node struct
    CK_Para.R = [];
    CK_Para.L = [];
    CK_Para.C = zeros(Nn,Nn);
    CK_Para.G = zeros(Nn,Nn);
    CK_Para.P = [];
    CK_Para.Cw = [];
    CK_Para.Ht = [];

    CK_Para.Vs = [];
    CK_Para.Is = [];
    CK_Para.Nle = [];
    CK_Para.Swh = [];
end
%--------------------------------------------------------------------------
% (4) Update CK_Para, Node, Bran, Meas with Aie2Gnd Bridge
[CK_Para,Bran,Blok] = Tower_A2G_Bridge(CK_Para,Node,Bran,Blok);  

% (5)Build the tower
Tower.CK_Para = CK_Para;               
Tower.Blok=Blok;
Tower.Bran=Bran;
Tower.Node=Node;
Tower.Meas=Meas;
% *********************** updated in March 2024****************************
Tower.Tower0 = Tower;                   % for updating CK model only
% *************************************************************************
Tower.WireP = WireP;
Tower.T2Smap = Tower_Map_Init();        % Mapping table initilization
Tower.T2Cmap = Tower_Map_Init();        % Mapping table initilization
Tower.Soc = [];

% (6) Updatee the tower with lump CK+    
Tower = Tower_Circuit_Update(Tower);

%-------------------------------------------------------------------------
% % (6) Updatee the tower with lump CK
% % (6a) Read Circuit Modules
% Bflag = Blok.flag;
% Bname = Blok.name;
% [CKins,Nins,Bins,Mins]=Lump_Model_Intepret(Bflag,Bname,2,Blok.ins);   
% [CKsar,Nsar,Bsar,Msar]=Lump_Model_Intepret(Bflag,Bname,3,Blok.sar);   
% [CKtxf,Ntxf,Btxf,Mtxf]=Lump_Model_Intepret(Bflag,Bname,4,Blok.txf);   
% [CKgrd,Ngrd,Bgrd,Mgrd]=Lump_Model_Intepret(Bflag,Bname,5,Blok.grd);   
% [CKint,Nint,Bint,Mint]=Lump_Model_Intepret(Bflag,Bname,6,Blok.int);   
% [CKinf,Ninf,Binf,Minf]=Lump_Model_Intepret(Bflag,Bname,7,Blok.inf);   
% [CKmck,Nmck,Bmck,Mmck]=Lump_Model_Intepret(Bflag,Bname,8,Blok.mck);   
% [CKoth1,Noth1,Both1,Moth1]=Lump_Model_Intepret(Bflag,Bname,9,Blok.oth1);   
% [CKoth2,Noth2,Both2,Moth2]=Lump_Model_Intepret(Bflag,Bname,10,Blok.oth2);   
% 
% % (6b) Update Tower_CK_Block
% [CK_Para,Node,Bran,Meas]=Tower_CK_Update(CK_Para,Node,Bran,Meas,...
%     CKins,Nins,Bins,Mins,Bflag(2));
% [CK_Para,Node,Bran,Meas]=Tower_CK_Update(CK_Para,Node,Bran,Meas,...
%     CKsar,Nsar,Bsar,Msar,Bflag(3));
% [CK_Para,Node,Bran,Meas]=Tower_CK_Update(CK_Para,Node,Bran,Meas,...
%     CKtxf,Ntxf,Btxf,Mtxf,Bflag(4));
% [CK_Para,Node,Bran,Meas]=Tower_CK_Update(CK_Para,Node,Bran,Meas,...
%     CKgrd,Ngrd,Bgrd,Mgrd,Bflag(5));
% [CK_Para,Node,Bran,Meas]=Tower_CK_Update(CK_Para,Node,Bran,Meas,...
%     CKint,Nint,Bint,Mint,Bflag(6));
% [CK_Para,Node,Bran,Meas]=Tower_CK_Update(CK_Para,Node,Bran,Meas,...
%     CKinf,Ninf,Binf,Minf,Bflag(7));
% [CK_Para,Node,Bran,Meas]=Tower_CK_Update(CK_Para,Node,Bran,Meas,...
%     CKmck,Nmck,Bmck,Mmck,Bflag(8));
% [CK_Para,Node,Bran,Meas]=Tower_CK_Update(CK_Para,Node,Bran,Meas,...
%     CKoth1,Noth1,Both1,Moth1,Bflag(9));
% [CK_Para,Node,Bran,Meas]=Tower_CK_Update(CK_Para,Node,Bran,Meas,...
%     CKoth2,Noth2,Both2,Moth2,Bflag(10));
% 
% % Add small C to diag. elements in C matrix -------------------------------
% for ik = 1:Node.num(1)
%     if CK_Para.C(ik,ik) == 0
%         CK_Para.C(ik,ik) = Cmin;
%     end
% end
% 
% % Delete psuedo bran in A (all-zero row)
% if Blok.flag(1)==0  
%     row_all_zeros=find(all(CK_Para.A == 0,2));
%     CK_Para.A(row_all_zeros,:) = [];
% end
% % -------------------------------------------------------------------------
% Tower.CK_Para=CK_Para;
% Tower.WireP=WireP;
% Tower.Bran=Bran;
% Tower.Node=Node;
% Tower.Meas=Meas;
% Tower.T2Smap = Tower_Map_Init();        % Mapping table initilization
% Tower.T2Cmap = Tower_Map_Init();        % Mapping table initilization
% Tower.Soc = [];
end
