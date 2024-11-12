function [Uout]=U_Cal_3(Er_out,Ez_out,pt_start,pt_end,Nt,x_hit,y_hit)
a00=size(pt_start,1);
% x_hit = 0;
% y_hit = 0;

Rx = (pt_start(:,1)./2+pt_end(:,1)./2-x_hit);
Ry = (pt_start(:,2)./2+pt_end(:,2)./2-y_hit);
Rxy = sqrt( Rx.^2 + Ry.^2 ) ;


for ik=1:a00
    x1 = pt_start(ik,1);
    y1 = pt_start(ik,2);
    z1 = pt_start(ik,3);
    
    x2 = pt_end(ik,1);
    y2 = pt_end(ik,2);
    z2 = pt_end(ik,3);
    if Rxy(ik)==0
            Uout(1:Nt,ik)=  Ez_out(1:Nt,ik)*(z1-z2);
    else
        Uout(1:Nt,ik)= Er_out(1:Nt,ik).*Rx(ik)/Rxy(ik)*(x1-x2) ...
        + Er_out(1:Nt,ik)*Ry(ik)/Rxy(ik)*(y1-y2) + Ez_out(1:Nt,ik)*(z1-z2);
%         Uout(1:Nt,ik)= Er_out(1:Nt,ik).*Rx(ik)/Rxy(ik) ...
%         + Er_out(1:Nt,ik)*Ry(ik)/Rxy(ik) + Ez_out(1:Nt,ik);
    end
end
end