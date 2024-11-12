function   con_id= Cond_ID_Read(Cir,Pos)
% Return Conductor ID of a Span given by Pos
%        Pos = [T/S/C, id, cir_id, phas_id, cond_id, seg_id]

cir_id = Pos(3);  % cir_id
pha_id = Pos(4);  % pha_id
if ismember(cir_id, Cir.dat(:,1))
    idx = find(Cir.dat(:,1)==cir_id);
    if idx == 1
        con_id = pha_id;             % id in the circuit group
    else
        con_id = sum(Cir.dat(1:idx-1,2),1)+pha_id;
    end
else
    error('ID in Global Data and Tower/Span Data does not match');
end
