function listdex = InfNodeUpdate(listdex, oftn)
% Reset id of 0 (oftn) to be 0 after mergerimng
% oftn

id_tmp = find(listdex==oftn);
listdex(id_tmp)= 0;
end