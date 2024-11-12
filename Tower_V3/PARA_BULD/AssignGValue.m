function G = AssignGValue(G,Glist)
% Assign Mod=1: potential coef (I=GV) of netowrk or
%        Mod=2: a voltage-control current source (Is=GVs)
% return G = val (+/-val) at pos (n1, n2) if n1/2~=0, and
%          = no change if n1/2 = 0

if isempty(Glist)
    return;
end

Nrow = size(Glist,1);
N1 = Glist(:,1);
N2 = Glist(:,2);
VAL = Glist(:,3);
MOD = Glist(:,4);

for i = 1:Nrow
    n1 = N1(i);
    n2 = N2(i);
    val = VAL(i);
    if MOD == 2
        if n1~=0
            G(n1,n1) = G(n1,n1)+val;
        end
        if n2~=0
            G(n2,n2) = G(n2,n2)+val;
        end
        if n1*n2~=0
            G(n1,n2) = G(n1,n2)-val;
            G(n2,n1) = G(n2,n1)-val;
        end
    elseif MOD ==1
        G(n1,n2) = G(n1,n2)+val;                    % controlled Is source
    end       
end
    