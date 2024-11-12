%  Function:       INT_SLAN_2D
%  Description:    Calculate intergral between two arbitrary lines according
%                  Inductance Calculations: working formulas and tables
%                  p.56
%  Calls:          INT_LINE_D2P(U1a,U1b,V1,W1,r1,U2a,U2b,V2,W2,r2)
%  Input:          ps1  --  coordinate of start point of source line
%                           [xs1 ys1 zs1]
%                  ps2  --  coordinate of end point of source line
%                           [xs2 ys2 zs2]
%                  l1   --  length of source line
%                  pf1  --  coordinate of start point of field line
%                  pf2  --  coordinate of end point of field line
%                  l2   --  length of field line
%                  PROD_MOD  --  1 for dot product, 2 for vector product
%                                out =  lf * ls'
%                  COEF_MOD  --  1 for potential (P), 2 for inductance (L), 
%                                out =  lf .* ls
%  Output:         int  --  integral result (always > 0)
%  Author:         Chen Hongcai
%  Email :         hc.chen@live.com
%  Date:           2015-12-13
%                  2019-11-15 by YD
% Note: Three speical cases 
%       (1) co-plane (D=0), 
%       (2) parallel (cos=1)
%       (3) the end point of a line on the other line (e.g., v=R1+R2)
%
% ps1=[0 0 0]; ps2=[2 0 0]; rs=1e-3; pf1=[1 0 0]; pf2=[1 1 0]; rf=1e-3; PROD_MOD=2; COEF_MOD=1;
% ps1=[0 0 0;0 0 0]; ps2=[2 0 0;0 -1 -1]; rs=1e-3; pf1=[0 1 0]; pf2=[1 1 0]; rf=1e-3; PROD_MOD=2; COEF_MOD=1;
%
function int = INT_SLAN_2D(ps1, ps2, rs, pf1, pf2, rf, PROD_MOD, COEF_MOD)
% (1) initialization
g0 = 1e-5;          % using formula for parallel lines 
d0 = 1e-6;          % on the same plane
r0 = 1e-10;         % to avoid 0 in integration/ parallel lines

[Ns n0] = size(ps1);
[Nf n0] = size(pf1);

ls2 = sum((ps1-ps2).*(ps1-ps2),2);
lf2 = sum((pf1-pf2).*(pf1-pf2),2);
ls = sqrt(ls2);
lf = sqrt(lf2);

% (2) determine the distance of 4 points
switch PROD_MOD
    case  1                      % dot product
        OMG = zeros(Nf,1);

R12 = (ps2(:,1)-pf2(:,1)).^2+(ps2(:,2)-pf2(:,2)).^2+(ps2(:,3)-pf2(:,3)).^2;
R22 = (ps2(:,1)-pf1(:,1)).^2+(ps2(:,2)-pf1(:,2)).^2+(ps2(:,3)-pf1(:,3)).^2;
R32 = (ps1(:,1)-pf1(:,1)).^2+(ps1(:,2)-pf1(:,2)).^2+(ps1(:,3)-pf1(:,3)).^2;
R42 = (ps1(:,1)-pf2(:,1)).^2+(ps1(:,2)-pf2(:,2)).^2+(ps1(:,3)-pf2(:,3)).^2;
    case 2                      % vector product
        OMG = zeros(Nf,Ns);

        ls = repmat(ls',Nf,1);
        lf = repmat(lf,1,Ns);
        ls2 = repmat(ls2',Nf,1);
        lf2 = repmat(lf2,1,Ns);

        dx=repmat(pf2(:,1),1,Ns)-repmat(ps2(:,1)',Nf,1);        
        dy=repmat(pf2(:,2),1,Ns)-repmat(ps2(:,2)',Nf,1);        
        dz=repmat(pf2(:,3),1,Ns)-repmat(ps2(:,3)',Nf,1);        
        R12 = dx.^2 + dy.^2 + dz.^2;
        
        dx=repmat(pf1(:,1),1,Ns)-repmat(ps2(:,1)',Nf,1);        
        dy=repmat(pf1(:,2),1,Ns)-repmat(ps2(:,2)',Nf,1);        
        dz=repmat(pf1(:,3),1,Ns)-repmat(ps2(:,3)',Nf,1);        
        R22 = dx.^2 + dy.^2 + dz.^2;

        dx=repmat(pf1(:,1),1,Ns)-repmat(ps1(:,1)',Nf,1);        
        dy=repmat(pf1(:,2),1,Ns)-repmat(ps1(:,2)',Nf,1);        
        dz=repmat(pf1(:,3),1,Ns)-repmat(ps1(:,3)',Nf,1);        
        R32 = dx.^2 + dy.^2 + dz.^2;
        
        dx=repmat(pf2(:,1),1,Ns)-repmat(ps1(:,1)',Nf,1);        
        dy=repmat(pf2(:,2),1,Ns)-repmat(ps1(:,2)',Nf,1);        
        dz=repmat(pf2(:,3),1,Ns)-repmat(ps1(:,3)',Nf,1);        
        R42 = dx.^2 + dy.^2 + dz.^2;
    otherwise
        disp('No such case in INT_ARBI_2D');
        return;
end
clear dx dy dz

R1 = sqrt(R12);             % pf2-ps2
R2 = sqrt(R22);             % pf1-ps2 
R3 = sqrt(R32);             % pf1-ps1 
R4 = sqrt(R42);             % pf2-ps1 

% (3) find the cos and sin
a2 = (R42-R32+R22-R12);

cose =a2./(2*ls.*lf);
sine2 = 1-cose.^2;
sine = sqrt(sine2);

% (3a) update u (alpha) and v (beta)
DIS = 4*ls2.*lf2-a2.*a2;
par1 = cose>1-g0;                                     % parallel lines 1
par2 = cose<g0-1;                                     % parallel lines 2
para = par1 | par2;
sign = par1 - par2;
sign = sign(para);                                    % sign for lf

u = (ls.*( (2*lf2.*(R22-R32-ls2)+a2.*(R42-R32-lf2))./DIS));
v = (lf.*( (2*ls2.*(R42-R32-lf2)+a2.*(R22-R32-ls2))./DIS));
u(para) = 0;
v(para)= -(R22(para)-R32(para)-ls2(para))./(2*ls(para));   % parallel lines

d2 = abs(R32-u.^2-v.^2+2*u.*v.*cose);
d = (sqrt(d2));
clear DIS a2 tp1 tp2 subs tp0 par1 par2

copn = d<d0;        % co-plane
Itmp = para|copn;   % both co-plane and parallel lines
id   = ~Itmp;       % lines other than Itmp
clear copn Itmp

R1 = max(r0,R1);    % avoid zero distance
R2 = max(r0,R2);
R3 = max(r0,R3);
R4 = max(r0,R4);

% (4) lines in different planes and non parallel ines
OMG(id) = atan((d2(id).*cose(id)+(u(id)+ls(id)).*(v(id)+lf(id)).*sine2(id))./(d(id).*R1(id).*sine(id))) ...
    - atan((d2(id).*cose(id)+(u(id)+ls(id)).*v(id).*sine2(id))./(d(id).*R2(id).*sine(id))) ...
    + atan((d2(id).*cose(id)+(u(id).*v(id)).*sine2(id))./(d(id).*R3(id).*sine(id))) ...
    - atan((d2(id).*cose(id)+u(id).*(v(id)+lf(id)).*sine2(id))./(d(id).*R4(id).*sine(id)));

% (5) main item (end pt positioned on the another line)
int = 0;
tp0 = lf./(R1+R2);
tp1 = (u+ls).*atanh(tp0);
tp1 (abs(tp0-1)<r0) = 0;
int = int + tp1;
 
tp0 = ls./(R1+R4);
tp1 = (v+lf).*atanh(tp0);
tp1 (abs(tp0-1)<r0) = 0;
int = int + tp1;

tp0 = lf./(R3+R4);
tp1 = u.*atanh(tp0);
tp1 (abs(tp0-1)<r0) = 0;
int = int - tp1;

tp0 = ls./(R2+R3);
tp1 = v.*atanh(tp0);
tp1 (abs(tp0-1)<r0) = 0;
int = int - tp1;
clear tp0 tp1 

tp0 = OMG.*d./sine;
tp0(abs(sine)<g0) = 0;
int = 2*int - tp0 ;

% int =  ( 2*( (u+ls).*atanh(lf./(R1+R2)) + (v+lf).*atanh(ls./(R1+R4)) ...
%     - u.*atanh(lf./(R3+R4)) - v.*atanh(ls./(R2+R3)) ) - OMG.*d./sine) ;
clear R12 R22 R32 R42 OMG d2 id_dp id_sp d2 ls2 lf2 tp0 tp1

% (6) update integral with parallel line results
if ~isempty(para)
    tp = zeros(Nf,Ns);
    if length(rf)==1
        Rf = repmat(rf,Nf,Ns);
    else
        Rf = repmat(rf,1,Ns);
    end
    if length(rs)==1
        Rs = repmat(rs,Nf,Ns);
    else
        Rs = repmat(rs',Nf,1);
    end
out = INT_LINE_D2P_D(tp(para),ls(para),       tp(para),0,Rs(para),...
                     v(para),v(para)+sign.*lf(para),d(para), 0,Rf(para));
int(para) = abs(out);
end

% (7) check whether it is the integral for inductance or potential
if COEF_MOD==2                   % inductance
    int = cose .* int;
end

end


