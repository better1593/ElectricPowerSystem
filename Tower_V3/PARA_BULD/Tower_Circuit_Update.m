function Tower = Tower_Circuit_Update(Tower)

Cmin = 1e-12;     
% *********************** updated in March 2024****************************
Tower0 = Tower.Tower0;
CK_Para = Tower0.CK_Para;                   
GND = Tower0.GND;
Blok = Tower0.Blok;
Bran = Tower0.Bran;
Node = Tower0.Node;
Meas = Tower0.Meas;

Tower.Blok = Blok;
Tower.GND = GND;
% *********************** updated in March 2024****************************

%-------------------------------------------------------------------------
% (5a) Read Circuit Modules
Bflag = Blok.flag;
Bname = Blok.name;
[CKins,Nins,Bins,Mins]=Lump_Model_Intepret(Bflag,Bname,2,Blok.ins);   
[CKsar,Nsar,Bsar,Msar]=Lump_Model_Intepret(Bflag,Bname,3,Blok.sar);   
[CKtxf,Ntxf,Btxf,Mtxf]=Lump_Model_Intepret(Bflag,Bname,4,Blok.txf);   
[CKgrd,Ngrd,Bgrd,Mgrd]=Lump_Model_Intepret(Bflag,Bname,5,Blok.grd);   
[CKint,Nint,Bint,Mint]=Lump_Model_Intepret(Bflag,Bname,6,Blok.int);   
[CKinf,Ninf,Binf,Minf]=Lump_Model_Intepret(Bflag,Bname,7,Blok.inf);   
[CKmck,Nmck,Bmck,Mmck]=Lump_Model_Intepret(Bflag,Bname,8,Blok.mck);   
[CKoth1,Noth1,Both1,Moth1]=Lump_Model_Intepret(Bflag,Bname,9,Blok.oth1);   
[CKoth2,Noth2,Both2,Moth2]=Lump_Model_Intepret(Bflag,Bname,10,Blok.oth2);   

% (5b) Update Tower_CK_Block
[CK_Para,Node,Bran,Meas]=Tower_CK_Update(CK_Para,Node,Bran,Meas,...
    CKins,Nins,Bins,Mins,Bflag(2));
[CK_Para,Node,Bran,Meas]=Tower_CK_Update(CK_Para,Node,Bran,Meas,...
    CKsar,Nsar,Bsar,Msar,Bflag(3));
[CK_Para,Node,Bran,Meas]=Tower_CK_Update(CK_Para,Node,Bran,Meas,...
    CKtxf,Ntxf,Btxf,Mtxf,Bflag(4));
[CK_Para,Node,Bran,Meas]=Tower_CK_Update(CK_Para,Node,Bran,Meas,...
    CKgrd,Ngrd,Bgrd,Mgrd,Bflag(5));
[CK_Para,Node,Bran,Meas]=Tower_CK_Update(CK_Para,Node,Bran,Meas,...
    CKint,Nint,Bint,Mint,Bflag(6));
[CK_Para,Node,Bran,Meas]=Tower_CK_Update(CK_Para,Node,Bran,Meas,...
    CKinf,Ninf,Binf,Minf,Bflag(7));
[CK_Para,Node,Bran,Meas]=Tower_CK_Update(CK_Para,Node,Bran,Meas,...
    CKmck,Nmck,Bmck,Mmck,Bflag(8));
[CK_Para,Node,Bran,Meas]=Tower_CK_Update(CK_Para,Node,Bran,Meas,...
    CKoth1,Noth1,Both1,Moth1,Bflag(9));
[CK_Para,Node,Bran,Meas]=Tower_CK_Update(CK_Para,Node,Bran,Meas,...
    CKoth2,Noth2,Both2,Moth2,Bflag(10));

% Add small C to diag. elements in C matrix -------------------------------
for ik = 1:Node.num(1)
    if CK_Para.C(ik,ik) == 0
        CK_Para.C(ik,ik) = Cmin;
    end
end

% Update Ground Conductance G
[CK_Para.G, CK_Para.P]= Ground_Potential(CK_Para,Node,GND);

% Delete psuedo bran in A (all-zero row)
if Blok.flag(1)==0  
    row_all_zeros=find(all(CK_Para.A == 0,2));
    CK_Para.A(row_all_zeros,:) = [];
end

% group meas according to its flag=1(I), 2(V), 3(I/V), 4(P), 5(V/I/P) 11(E)
mrow = size(Meas.flag);
Meas.Ib = [];  
Meas.Vn = [];   
Meas.Pw = [];  
Meas.En = [];   
Meas.IbList= [];
Meas.VnList= [];
Meas.PwList= [];
Meas.EnList= [];
for ik = 1:mrow
    switch Meas.flag(ik)
        case 1                                  % current
            Meas.Ib =[Meas.Ib; Meas.listdex(ik,1)];
            Meas.IbList =[Meas.IbList; Meas.list(ik,1)];
        case 2                                  % voltage
            Meas.Vn =[Meas.Vn; Meas.listdex(ik,2:3)];
            Meas.VnList =[Meas.VnList; Meas.list(ik,2:3)];
        case 3                                  % power
            Meas.Pw=[Meas.Pw; Meas.listdex(ik,:)];
            Meas.PwList =[Meas.PwList; Meas.list(ik,:)];
        case 4                                  % power, current, votlage
            Meas.Ib =[Meas.Ib; Meas.listdex(ik,1)];
            Meas.IbList =[Meas.IbList; Meas.list(ik,1)];
            Meas.Vn =[Meas.Vn; Meas.listdex(ik,2:3)];
            Meas.VnList =[Meas.VnList; Meas.list(ik,2:3)];
            Meas.Pw=[Meas.Pw; Meas.listdex(ik,:)];
            Meas.PwList =[Meas.PwList; Meas.list(ik,:)];
        case 11                                  % Energy
            Meas.En=[Meas.En; Meas.listdex(ik,:)];
            Meas.EnList =[Meas.EnList; Meas.list(ik,:)];
        otherwise
            disp('!!! Not used in Tower Circuit Update (Meas)');
    end
end
% -------------------------------------------------------------------------
Tower.CK_Para=CK_Para;
Tower.Bran=Bran;
Tower.Node=Node;
Tower.Meas=Meas;
end
