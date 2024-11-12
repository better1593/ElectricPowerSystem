function [Er_lossy]= Above_lossy(HR0,ER,GLB,sigma0)

erg=GLB.GND.epr;
sigma_g=GLB.GND.sig;
if isempty(sigma0)==0
    sigma_g=sigma0;
end
dt=GLB.dT;
Nt=GLB.Nt;



ep0=8.85e-12;
u0=4*pi*1e-7;
Nt0=Nt;
vc=3e8;
Nd=9;    
w=[0.01 0.05 0.1 0.5 1e0 5e0 1e1 5e1 1e2 5e2 1e3 5e3 1e4 5e4 1e5 5e5 1e6 5e6 1e7];
con=1;
a11=length(w);
H_in=zeros(1,1,a11);
for ii=1:a11
    H_in(ii)=vc*u0/sqrt(erg+sigma_g/(j*w(ii)*ep0));
end

% test1=zeros(a11,1);
% test1(1:a11)=H_in(1,1,1:a11);
[R0, L0, Rn, Ln, Zfit] = vecfit_kernel_Z_Ding(H_in, w/2/pi, Nd);
% test11=zeros(a11,1);
% test11(1:a11)=Zfit(1,1,1:a11);
R0_1=R0-sum(Rn,3);
L0_1=L0;
R_1=zeros(1,Nd);
R_1(1:Nd)=Rn(1,1,1:Nd);
L_1=zeros(1,Nd);
L_1(1:Nd)=Rn(1,1,1:Nd);

[a00 Nt]=size(HR0);
t_ob=Nt*dt;
conv_2=2;
dt0=dt/conv_2;
Nt0=Nt;
Nt3=Nt;
dt0=dt/conv_2;

    x = dt:dt:t_ob;     
y = HR0(:,1:Nt); 
xi = dt0:dt0:t_ob; 
H_save2 = (interp1(x,y',xi,'spline'))';
H_all=H_save2;
if a00==1
   H_all=H_all';
end

% H_all=HR0;
Ntt=size(H_all,2);  
H_all_diff=zeros(a00,Ntt);
H_all_diff(:,1)=H_all(:,1)/dt0;
H_all_diff(:,2:Ntt)=(diff(H_all')/dt0)';

ee0=R0.*H_all;
eeL=L0.*H_all_diff;


t00=Ntt;
ee=zeros(Ntt,Nd);
Rn2(1,1:Nd)=Rn(1,1,1:Nd);
Ln2(1,1:Nd)=Ln(1,1,1:Nd);
Rn3=repmat(Rn2,t00,1);
Ln3=repmat(Ln2,t00,1);
ee2=zeros(t00,Nd);
tt00=ones(t00,Nd);
tt00(1:t00,:)=repmat((1:t00)',1,Nd);
ee(1:t00,1:Nd)=-Rn3(1:t00,1:Nd).^2./Ln3(1:t00,1:Nd).*exp(-Rn3(1:t00,1:Nd)./Ln3(1:t00,1:Nd).*tt00.*dt0);
% for ii=1:t00
%     for jj=1:Nd
%         ee(ii,jj)=-Rn(1,1,jj)^2/Ln(1,1,jj)*exp(-Rn(1,1,jj)/Ln(1,1,jj)*(ii)*dt0);
%     end
% end
% ee_conv(:,:,1:Nd)=dt0*conv2(H_all,ee(1:Ntt,1:Nd)');


for jj=1:Nd
    ee_conv(:,:,jj)=dt0*conv2(H_all,ee(1:Ntt,jj)');
end

ee_conv_sum=sum(ee_conv,3);
ee_all=ee0(:,1:conv_2:Ntt)+eeL(:,1:conv_2:Ntt)+ee_conv_sum(:,1:conv_2:Ntt);
Er_lossy=ER+ee_all';

