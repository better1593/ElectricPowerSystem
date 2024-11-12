
function [U]= Cor_Lossy_ground(GLB, LGT, Lne,Er_T,Ez_T)

% addpath('Vectorfitting');

[H_p]  =H_Cal(GLB, LGT, Lne);
[Er_lossy]= Above_lossy(H_p,Er_T,GLB);

Ez_lossy=Ez_T;
E_T = sqrt(Er_lossy.^2+Ez_lossy.^2);

L=Lne.tran.L';
Nt=GLB.Nt;
for ia=2:Nt
L(ia,:)=L(1,:);
end

U= E_T .* L;
end