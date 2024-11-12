function Elem= AssignElemID(Ref_Elem,Elem,rng)
% Update Elem.listdex(:,rng) according to Elem.list using Ref_Elem.list
% 
% Elem     = Note or Bran, 
% rng      = col # to be updated
% Ref_Elem = Note.list(:.1) or 
%            Bran.list(:.1)

Nrow = size(Elem.list,1);
Ncol = length(rng);
for i =1:Nrow
    for j =1:Ncol
        id = rng(j);
        str = Elem.list(i,id);
        row_id = find(Ref_Elem.list==str);
        if ~isempty(row_id)
            Elem.listdex(i,id) = Ref_Elem.listdex(row_id);
            if ~isempty(Ref_Elem.pos)
                Elem.pos(i,1:3)=Ref_Elem.pos(row_id,1:3);
            end
        end
    end
end
end