
function [U]= Cor_Lossy_ground(GLB, LGT, GND, Lne,Er_T,Ez_T)

% addpath('Vectorfitting');

[H_p]  =H_Cal(GLB, LGT, Lne);
[Er_lossy]= Above_lossy(-H_p, Er_T, GLB, GND.sig); % fixed
% Er_lossy=Er_T;

Ez_lossy=Ez_T;
% E_T = sqrt(Er_lossy.^2+Ez_lossy.^2);

L=Lne.tran.L';
Nt=GLB.Nt;
for ia=2:Nt
L(ia,:)=L(1,:);
end

pt_start=Lne.tran.pt_start;
pt_end=Lne.tran.pt_end;
Nt=GLB.Nt;
[Uout]=U_Cal3(Er_lossy,Ez_lossy,pt_start,pt_end,Nt,LGT.Lch.pos(1),LGT.Lch.pos(2));
U=Uout;
% U= E_T .* L;
end