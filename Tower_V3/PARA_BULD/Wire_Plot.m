function Wire_Plot(WireP,NodeW,BranW)
% (1) DRAW LINE DIAGRAM 
scale=0.1;
wireplot=1;                     % plot wire yes=1 no=0;

n1=WireP(:,15);
n2=WireP(:,16);
str1=BranW.list(:,2);% str1=num2str(n1);
str2=BranW.list(:,3);% str2=num2str(n2);

x1=WireP(:,1);y1=WireP(:,2);z1=WireP(:,3);
x2=WireP(:,4);y2=WireP(:,5);z2=WireP(:,6);
px=[x1 x2]; dx=x2-x1;
py=[y1 y2]; dy=y2-y1;
pz=[z1 z2]; dz=z2-z1;
px0=0.5*(x1+x2);
py0=0.5*(y1+y2);
pz0=0.5*(z1+z2);
ds=sqrt(dx.*dx+dy.*dy+dz.*dz);
cosa=dx./ds;
cosb=dy./ds;
cosc=dz./ds;

plot3(px',py',pz','-','LineWidth',2);hold on;
quiver3(x1,y1,z1,cosa,cosb,cosc,scale,'r','fill');
text(x1,y1,z1,str1,'FontSize',12,'Color','red');
text(x2,y2,z2,str2,'FontSize',12,'Color','red');

% for ik=1:length(NodeW.comdex)
%     str=NodeW.com(ik);
%     I0=NodeW.comdex(ik);
%     tmp=find(n1==I0);
%     if ~isempty(tmp)
%         pos=tmp(1);
%         text(x1(pos),y1(pos),z1(pos),str,'FontSize',12,'Color','blue'); 
%     end
%     tmp=find(n2==I0);
%     if ~isempty(tmp)
%        pos=tmp(1);
%        text(x2(pos),y2(pos),z2(pos),str,'FontSize',12,'Color','blue'); 
%     end
% end
% 
% for ik=1:length(NodeR.comdex)
%     str=NodeR.com(ik);
%     I0=NodeR.comdex(ik);
%     tmp=find(n1==I0);
%     if ~isempty(tmp)
%         pos=tmp(1);
%         text(x1(pos),y1(pos),z1(pos),str,'FontSize',12,'Color','blue'); 
%     end
%     tmp=find(n2==I0);
%     if ~isempty(tmp)
%         pos=tmp(1);
%         text(x2(pos),y2(pos),z2(pos),str,'FontSize',12,'Color','blue'); 
%     end
% end
% 
% for ik=1:length(NodeS.comdex)
%     str=NodeS.com(ik);
%     I0=NodeS.comdex(ik);
%     tmp=find(n1==I0);
%     if ~isempty(tmp)
%         pos=tmp(1);
%         text(x1(pos),y1(pos),z1(pos),str,'FontSize',12,'Color','blue'); 
%     end
%     tmp=find(n2==I0);
%     if ~isempty(tmp)
%         pos=tmp(1);
%         text(x2(pos),y2(pos),z2(pos),str,'FontSize',12,'Color','blue'); 
%     end
% end
% 
% for ik=1:length(NodeX.comdex)
%     str=NodeX.com(ik);
%     I0=NodeX.comdex(ik);
%     tmp=find(n1==I0);
%     if ~isempty(tmp)
%         pos=tmp(1);
%         text(x1(pos),y1(pos),z1(pos),str,'FontSize',12,'Color','blue'); 
%     end
%     tmp=find(n2==I0);
%     if ~isempty(tmp)
%         pos=tmp(1);
%         text(x2(pos),y2(pos),z2(pos),str,'FontSize',12,'Color','blue'); 
%     end
% end
% 
% if ~isempty(NodeG)
% for ik=1:length(NodeG.comdex)
%     str=NodeG.com(ik);
%     I0=NodeG.comdex(ik);
%     tmp=find(n1==I0);
%     if ~isempty(tmp)
%         pos=tmp(1);
%         text(x1(pos),y1(pos),z1(pos),str,'FontSize',12,'Color','blue'); 
%     end
%     tmp=find(n2==I0);
%     if ~isempty(tmp)
%         pos=tmp(1);
%         text(x2(pos),y2(pos),z2(pos),str,'FontSize',12,'Color','blue'); 
%     end
% end

end