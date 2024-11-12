function Posi = LGTS_Pos_Update(Span,Posi)
% Update Soc.pos when a tower is struck directly given by Span(Posi(2))
% pos = [T, (Tower_ID), Cir_id, pha_id, (Node_ID) Flag=1]  

if Posi(1)~=1 || Span.ID~=Posi(2) || Posi(6)==1
    return;     % return for non-direct-strike to tower or non-rel. span
end

cir_id = Posi(3);
pha_id = Posi(4);
seg_id = Posi(5);

Info = Span.Info;
S2Tmap = Span.S2Tmap;
Cir = Span.Cir;

con_id = Cond_ID_Read(Cir,Posi);
hid = Info{1,5};
tid = Info{1,6};

if seg_id == 1
    Tower_ID = hid;
    map = S2Tmap.head(2:end,:);
else
    Tower_ID = tid;
    map = S2Tmap.tail(2:end,:);
end    
Node_ID = map(con_id,2);

Posi(2) = Tower_ID;
Posi(5) = Node_ID;
Posi(6) = 1;
end


