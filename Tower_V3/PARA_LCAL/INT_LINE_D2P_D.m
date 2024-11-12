%INT_LINE_D2P_D   Inductance function (Neuuman formula) for 
%                   wire-to-wire (source-to-obervation)
%                   subscript "1" = observation, "2" = source 
%
%          DOT PRODUCT of Two Vectors
%
%          Inegration of (1/sqrt((u1-u2)^2+As) over du1 & du2 
%          As=(v1-v2)^2+dw.*dw, u1 and u2 = wire direction
%
% U1a U1b  V1, W1    lower and upper limtis of field lines (vector)
% U2a U2b  V2, W2    lower and upper limtis of source lines (vector)
% r1, r2  > ELIM     radius of two sets of filaments
%                    =wire radius for wires
% out:               coef. of zero order approximation
%
%          Revised on 31/12/2014

function out=INT_LINE_D2P_D(U1a,U1b,V1,W1,r1,U2a,U2b,V2,W2,r2)
ELIM=1e-9;                              % LIMIT FOR CHANGING FORMULA
a2=max(r2,r1); % to avoid log(0) for negative uij
a2=a2.*a2;

no=length(U1a);
ns=length(U2a);
if no~=ns out=[]; return; end

u13=U1a-U2a;
u14=U1a-U2b;
u23=U1b-U2a;
u24=U1b-U2b;

u13s=u13.*u13;  
u23s=u23.*u23;  
u14s=u14.*u14;  
u24s=u24.*u24;  

As=(V2-V1).*(V2-V1)+(W2-W1).*(W2-W1);
As=max(As,a2);
t13=sqrt(As+u13s); 
t23=sqrt(As+u23s);  
t14=sqrt(As+u14s);  
t24=sqrt(As+u24s);  

%using the exact formulas for calculation
I1=-u24.*log(u24+t24)-u13.*log(u13+t13)+u23.*log(u23+t23)+u14.*log(u14+t14);

s=u24+t24;Idex=s<ELIM;
s=u13+t13;
s=u23+t23;
s=u14+t14;
if sum(sum(Idex))~=0
    I1a=+u24.*log(t24-u24)+u13.*log(t13-u13)-u23.*log(t23-u23)-u14.*log(t14-u14);%+log(As).*(-u24-u13+u23+u14);
    I1(Idex)=I1a(Idex);
end

I2=t24+t13-t23-t14;
out=I1+I2;
end