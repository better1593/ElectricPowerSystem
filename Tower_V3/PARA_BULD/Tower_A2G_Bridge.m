function [CK_Para,Bran,Blok] = Tower_A2G_Bridge(CK_Para,Node,Bran,Blok)
% Building GRound surface model connecting air-node/gnd-node blocks

if Blok.flag(1)~=2
    return;
end

% (1) Initilization
Rmin = 1e-6;
Nn = Node.num(1);                               % total # of Wire Node
Nb = Bran.num(1);                               % total # of Wire Bran
a2g = Blok.a2g;                                 % Air-2-Gnd Bridge: 
list = a2g.list;                                % [b0 air_node gnd_node]
listdex = [];
Np = size(a2g.list,1);                          % # of A2G Bridge

A = CK_Para.A;
R = CK_Para.R;
L = CK_Para.L;

% (2) Updating Bran.list/listdex/num and Blok.agb
for i= 1:Np
    tmp1 = find(Node.list==list(i,2));
    tmp2 = find(Node.list==list(i,3));
    listdex = [listdex;[i+Nb, tmp1, tmp2]];
end
a2g.listdex = listdex;
Blok.a2g = a2g;

tmp0 =[Np 0 0 0 0 Np];
Bran.num = Bran.num + tmp0;
Bran.list = [Bran.list; list];
Bran.listdex = [Bran.listdex; listdex];

% (3) Update A, R and L
Asf = zeros(Np,Nn);
Rsf = eye(Np)*Rmin;
Lsf = zeros(Np,Np);
for i =1:Np
    Asf(i,listdex(i,2)) = -1;           % air -> gnd
    Asf(i,listdex(i,3)) = +1;           % air -> gnd
end
A = [A; Asf];
R = blkdiag(R,Rsf);
L = blkdiag(L,Lsf);

% (4) CK-Para Updating
CK_Para.A = A;
CK_Para.R = R;
CK_Para.L = L;
end