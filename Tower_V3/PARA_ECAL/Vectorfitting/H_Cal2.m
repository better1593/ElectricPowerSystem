function [Uout_total2]  =H_Cal2(pt_hit,h_ch,Ns_ch, pt_start,pt_end, i_sr, t_sr)


% function  sr_induced_v_num_1: calculate lightning induced E-filed using Jeffimenko's equations
% pt_hit=[0 0]; % strike location
% t_ch=0;  % useless
% Nt_ch=0; % useless
% pt_start=[49.5 0 6;59.5 0 6;]; % start point of line above ground
% pt_end  =[50.5 0 6;60.5 0 6;]; % end point of line above ground
% pt_start_grid=[];% start point of line below ground
% pt_end_grid=[]; % end point of line below ground
% sigma_soil=0.001; % soil conductivity
% i_sr=ones(1,100); % user-defined source
% t_sr=(1:100)*0.01; % user-define time step
pt_start_grid=[];
pt_end_grid=[];
% h_ch=2000; % height of lightning channel
if h_ch>2000
    Ns_ch=h_ch/1; % discretization of channel
else
    Ns_ch=h_ch/1; % discretization of channel
end

if max(abs(i_sr))==0
    Ns_ch=1;
end
dz_ch=h_ch/Ns_ch;

if isempty(pt_start_grid)==0;
Nc1a=size(pt_start,1);
Nc1b=size(pt_start_grid,1);
pt_start_0=zeros(Nc1a+Nc1b,3);
pt_end_0=zeros(Nc1a+Nc1b,3);
pt_start_0(1:Nc1a,:)=pt_start;
pt_start_0(Nc1a+(1:Nc1b),:)=pt_start_grid;
pt_end_0(1:Nc1a,:)=pt_end;
pt_end_0(Nc1a+(1:Nc1b),:)=pt_end_grid;


pt_start=pt_start_0;
pt_end=pt_end_0;
else
end

Nc1=size(pt_start,1);
for ia=1:Nc1
pt_start(ia,1:2)=pt_start(ia,1:2)+pt_hit;
pt_end(ia,1:2)=pt_end(ia,1:2)+pt_hit;
end

erg=10;
Nc1=size(pt_start,1);
[pt_a0 pt_b0]=size(pt_start);

flag_type=3;
Nt = length(t_sr);
ep0 = 8.85*1e-12;
dt = (t_sr(2)-t_sr(1))*1e-6;

if flag_type == 1  %% TL model
    vc = 3e8;
    ve=3e8;
    vcof = ve/vc;
    lamda=2e4;
elseif flag_type == 2  %% MTLL
    vc = 3e8;
    vcof = ve/vc;
    H = 7e3;
else %% MTLE
    vc = 3e8;
    ve=1e8;
    vcof = ve/vc;
    lamda = 1.7e3;  % constant in MTLE -- decays exponentially with the height
end
i_sr_int = zeros(Nt,1);
i_sr_div = zeros(Nt,1);
i_sr0=i_sr;

if size(i_sr0,1)==1
    i_sr0 = i_sr0';
end
i_sr_int(1:Nt) = cumsum(i_sr0(1:Nt))*dt;
i_sr_div(2:Nt) = diff(i_sr0(1:Nt))/dt;
i_sr_div(1)=i_sr0(1)/dt;

for ia=1:Nt
    for ib=1:Ns_ch
        td=round(((ib-1)*dz_ch+0.5*dz_ch)/ve/dt);
        z_ch=(ib-1)*dz_ch;
        q_sr_a(ib,td+ia)=1/lamda.*exp(-z_ch/lamda).*i_sr_int(ia)+exp(-z_ch/lamda)*i_sr0(ia)/vc/vcof;
    end
end

for ia=1:Nt
    for ib=1:Ns_ch
        td=round(((ib-1)*dz_ch+0.5*dz_ch)/ve/dt);
        z_ch=(ib-1)*dz_ch;
        i_sr_a(ib,td+ia)=exp(-z_ch/lamda)*i_sr0(ia);
    end
end

q_sr_a2=q_sr_a(1:Ns_ch,1:Nt);
i_sr_a2=i_sr_a(1:Ns_ch,1:Nt);



abc=1;
nk=abc;
for i=1:pt_a0
Kx=(pt_end(i,1)-pt_start(i,1))/nk;
Ky=(pt_end(i,2)-pt_start(i,2))/nk;
Kz=(pt_end(i,3)-pt_start(i,3))/nk;
Point((nk+1)*(i-1)+1,:)=pt_start(i,:);
for ik=1:1*nk
    Point((nk+1)*(i-1)+ik+1,1)=Point((nk+1)*(i-1)+ik,1)+Kx;
    Point((nk+1)*(i-1)+ik+1,2)=Point((nk+1)*(i-1)+ik,2)+Ky;
    Point((nk+1)*(i-1)+ik+1,3)=Point((nk+1)*(i-1)+ik,3)+Kz;
end
end
pt_start2=zeros(pt_a0*abc,pt_b0);
for i=1:pt_a0
pt_start2((abc*(i-1)+1):(abc*i),:)=Point(((abc+1)*(i-1)+1):((abc+1)*i-1),:);
pt_end2((abc*(i-1)+1):(abc*i),:)=Point(((abc+1)*(i-1)+2):((abc+1)*i),:);
end

pt_start_img = pt_start2;
pt_end_img = pt_end2;
pt_start_img(:,3) = -(pt_start2(:,3));
pt_end_img(:,3) = -(pt_end2(:,3));

Nc=size(pt_start2,1); % 
[pt_a pt_b]=size(pt_start2);

pt_start0=pt_start2;
pt_end0=pt_end2;
for i=1:Nc
    pt_start0(i,3)=0;
    pt_end0(i,3)=0;
end

a00=Nc;

dz_ch = h_ch/Ns_ch;

Uout0 = zeros(Nt,Nc);
Er_T = zeros(Nt,Nc);
Ez_T = zeros(Nt,Nc);

R_xy0=0;
for ia=1:Nc
    R_xy0(ia)=sqrt((pt_start(ia,1)/2+pt_end(ia,1)/2)^2+(pt_start(ia,2)/2+pt_end(ia,2)/2)^2);
end
R_avg=sum(R_xy0)/Nc;
td0=floor(R_avg/3e8/dt);


for ik=1:a00
    
for ig=1:Ns_ch
    z_ch=ig*dz_ch-dz_ch/2;
[Uout_total(ig,:),Er_total(ig,:),Ez_total(ig,:)] = E_cal_J2_q(pt_start2(ik,:),pt_end2(ik,:),i_sr_a2(ig,1:Nt),t_sr,z_ch);
end
Er_total2(ik,:)=ep0*sum(Er_total,1);
Ez_total2(ik,:)=ep0*sum(Ez_total,1);
Uout_total2(ik,:)=2*dz_ch*ep0*sum(Uout_total,1);

end

