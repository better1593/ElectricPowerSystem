function C = AssignCValue(C,Clist)
% return C = val (+/-val) at pos (n1, n2) if n1/2~=0, and
%          = no change if n1/2 = 0
if isempty(Clist)
    return;
end

Nrow = size(Clist,1);
N1 = Clist(:,1);
N2 = Clist(:,2);
VAL = Clist(:,3);

for i = 1:Nrow
    n1 = N1(i);
    n2 = N2(i);
    val = VAL(i);
    if n1~=0
        C(n1,n1) = C(n1,n1)+val;
    end
    if n2~=0
        C(n2,n2) = C(n2,n2)+val;
    end
    if n1*n2~=0
        C(n1,n2) = C(n1,n2)-val;
        C(n2,n1) = C(n2,n1)-val;
    end
end   
end