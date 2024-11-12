function A = AssignAValue(A,b1,n1,val)
% return A = val (+/-1) at pos (b1, n1) if n1~=0, and
%          = no change if n1 = 0
if n1~=0
    A(b1,n1) = val;
end
end
    