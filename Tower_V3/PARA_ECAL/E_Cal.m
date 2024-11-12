
function [Er_T, Ez_T] = E_Cal(LGT, Lne)
    ep0 = 8.85e-12;
    vc = 3e8; % light speed
    
    % Extracting variables from input structures
    
    i_sr = LGT.Lch.curr;
    dt = LGT.Lch.dT;
    pt_start = [Lne.x1 Lne.y1 Lne.z1];
    pt_end = [Lne.x2 Lne.y2 Lne.z2];
%---------------debug
    % pt_start(1:2,1:3)=[199.5 0 0; 499.5 0 0];
    % pt_end(1:2,1:3)=[200.5 0 0; 500.5 0 0];
% 

%---------------
    Ns_ch = LGT.Lch.Nc;
    dz_ch = LGT.Lch.dH; 
    pt_hit = LGT.Lch.pos;
    flag_type = LGT.Lch.flg;
    H = LGT.Lch.H0;
    lamda = LGT.Lch.lam;
    vcof = 1/LGT.Lch.vcf;
    vcof=1.1/3;
    lamda=2e13;
% vcof is the speed of channel current over light speed
Nt=length(i_sr);  
t_sr=(1:Nt)*dt*1e6;   % dt单位us

x_hit=pt_hit(1);
y_hit=pt_hit(2);

z_ch = ((1:Ns_ch)-0.5)'*dz_ch; % mid point of the channel segnt
z_ch_img = -z_ch; % mid point of the channel segment

Rx = (pt_start(:,1)/2+pt_end(:,1)/2-x_hit);
Ry = (pt_start(:,2)/2+pt_end(:,2)/2-y_hit);
Rxy = sqrt( Rx.^2 + Ry.^2 ) ;

i_sr_int = zeros(Nt,1);
i_sr_div = zeros(Nt,1);

i_sr_int(1:Nt) = cumsum(i_sr(1:Nt))*dt;
i_sr_div(2:Nt) = diff(i_sr(1:Nt))/dt;
i_sr_div(1)=i_sr(1)/dt;


a00=size(pt_start,1); % a00 number of observation point
Er_T=zeros(Nt,a00);
Ez_T=zeros(Nt,a00);
for ik=1:a00
    %ik
    x1 = pt_start(ik,1);
    y1 = pt_start(ik,2);
    z1 = pt_start(ik,3);
    
    x2 = pt_end(ik,1);
    y2 = pt_end(ik,2);
    z2 = pt_end(ik,3);

    dEz1_air = zeros(Nt,Ns_ch);
    dEz2_air = zeros(Nt,Ns_ch);
    dEz3_air = zeros(Nt,Ns_ch);
    dEr1_air = zeros(Nt,Ns_ch);
    dEr2_air = zeros(Nt,Ns_ch);
    dEr3_air = zeros(Nt,Ns_ch);

    Rxyz = zeros(1,Ns_ch);
    Rz = zeros(1,Ns_ch);

    for ig=1:Ns_ch
        
%     for ig=351:351

        Rxyz(ig) = sqrt( Rxy(ik)^2+(z1/2+z2/2 - z_ch(ig))^2 );
        
        n_td_tmp = floor( ((t_sr)*1e-6 - (z_ch(ig)/vc/vcof + Rxyz(ig)/vc) )/dt );   %% time delay
        id_t = n_td_tmp>0;        
    
        % propagate part
        Rz(ig) = (z1/2+z2/2 - z_ch(ig));
        
        if flag_type == 1  %% TL model
            cof_isr = 1/(4*pi*ep0);
        elseif flag_type == 2  %% MTLL
            cof_isr = 1/(4*pi*ep0) * (1 - z_ch(ig)/H);
            
        else %% MTLE
            cof_isr = 1/(4*pi*ep0) * exp(-z_ch(ig)/lamda);
        end
        
        dEz_1_cof = cof_isr * (2*Rz(ig)^2-Rxy(ik)^2)/(Rxyz(ig)^5);
        dEz_2_cof = cof_isr * (2*Rz(ig)^2-Rxy(ik)^2)/(Rxyz(ig)^4)/vc;
        dEz_3_cof = cof_isr * (Rxy(ik)^2)/(Rxyz(ig)^3)/(vc^2);
        
        dEr_1_cof = cof_isr * (3*Rz(ig)*Rxy(ik))/(Rxyz(ig)^5);
        dEr_2_cof = cof_isr * (3*Rz(ig)*Rxy(ik))/(Rxyz(ig)^4)/vc;
        dEr_3_cof = cof_isr * (Rz(ig)*Rxy(ik))/(Rxyz(ig)^3)/(vc^2); 
        
        dEz1_air(id_t,ig) = dEz_1_cof * i_sr_int(n_td_tmp(id_t));
        dEz2_air(id_t,ig) = dEz_2_cof * i_sr(n_td_tmp(id_t));
        dEz3_air(id_t,ig) = dEz_3_cof * i_sr_div(n_td_tmp(id_t));
        
        dEr1_air(id_t,ig) = dEr_1_cof * i_sr_int(n_td_tmp(id_t));
        dEr2_air(id_t,ig) = dEr_2_cof * i_sr(n_td_tmp(id_t));
        dEr3_air(id_t,ig) = dEr_3_cof * i_sr_div(n_td_tmp(id_t));
        
     end
    
    Ez_air = sum( dEz1_air + dEz2_air - dEz3_air, 2 );
    Er_air = sum( dEr1_air + dEr2_air + dEr3_air, 2 );

    %----------------------- img
    dEz1_img = zeros(Nt,Ns_ch);
    dEz2_img = zeros(Nt,Ns_ch);
    dEz3_img = zeros(Nt,Ns_ch);
    dEr1_img = zeros(Nt,Ns_ch);
    dEr2_img = zeros(Nt,Ns_ch);
    dEr3_img = zeros(Nt,Ns_ch);
    
    Rxyz_img = zeros(1,Ns_ch);
    Rz_img = zeros(1,Ns_ch);
    
    for ig = 1:Ns_ch
        
        Rxyz_img(ig) = sqrt(Rxy(ik)^2+(z1/2+z2/2 - z_ch_img(ig))^2);
        
        n_td_tmp = floor( (t_sr*1e-6 - (abs(z_ch_img(ig))/vc/vcof + Rxyz_img(ig)/vc) )/dt );
        id_t = n_td_tmp>0;
        

        Rz_img(ig) = (z1/2+z2/2-z_ch_img(ig));

        if flag_type == 1  %% TL model
            cof_isr = 1/(4*pi*ep0);
        elseif flag_type == 2  %% MTLL
            cof_isr = 1/(4*pi*ep0) * (1 + z_ch_img(ig)/H);
        else %% MTLE
            cof_isr = 1/(4*pi*ep0) * exp(-abs(z_ch_img(ig))/lamda);
        end
        
        dEz1_img_cof = cof_isr*(2*Rz_img(ig)^2-Rxy(ik)^2)/Rxyz_img(ig)^5;
        dEz2_img_cof = cof_isr*(2*Rz_img(ig)^2-Rxy(ik)^2)/Rxyz_img(ig)^4/vc;
        dEz3_img_cof = cof_isr*(Rxy(ik)^2)/Rxyz_img(ig)^3/(vc^2);
        
        dEr1_img_cof = cof_isr*(3*Rz_img(ig)*Rxy(ik))/Rxyz_img(ig)^5;
        dEr2_img_cof = cof_isr*(3*Rz_img(ig)*Rxy(ik))/Rxyz_img(ig)^4/vc;
        dEr3_img_cof = cof_isr*(Rz_img(ig)*Rxy(ik))/Rxyz_img(ig)^3/(vc^2);        
       
        dEz1_img(id_t,ig) = dEz1_img_cof * i_sr_int(n_td_tmp(id_t));
        dEz2_img(id_t,ig) = dEz2_img_cof * i_sr(n_td_tmp(id_t));
        dEz3_img(id_t,ig) = dEz3_img_cof * i_sr_div(n_td_tmp(id_t));
        
        dEr1_img(id_t,ig) = dEr1_img_cof * i_sr_int(n_td_tmp(id_t));
        dEr2_img(id_t,ig) = dEr2_img_cof * i_sr(n_td_tmp(id_t));
        dEr3_img(id_t,ig) = dEr3_img_cof * i_sr_div(n_td_tmp(id_t));
        

    end

    Ez_img = sum( dEz1_img + dEz2_img - dEz3_img, 2 );
    Er_img = sum( dEr1_img + dEr2_img + dEr3_img, 2 );
    
    %--------------------------------------------------------
    Er_T(1:Nt,ik) = dz_ch*(Er_air+Er_img);
    Ez_T(1:Nt,ik) = dz_ch*(Ez_air+Ez_img);
    
    % Er and Ez is determined by the equation used in reference, so...
%     E_T(1:Nt,ik) = sqrt(Er_T^2+Ez_T^2);  % value. the direction is parrallel to the line
      % make sure with Ding
%       pt_start = [10,20,5;];
%       pt_end = [70,20,5;];
      
      
end
a=0;
