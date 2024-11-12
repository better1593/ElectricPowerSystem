function  [NodeTower, oft]= AssignElemLump(NodeTower,NodeLump,oft)
% copy node of a lump element (NodeLump) to the node of a tower (NodeTower)
% NodeTower.list, listdex, pos = [0 0 0]
% oft = offset of node ID

if isempty(NodeLump)
    return;
end

NT=NodeTower;
tmp =[0 0 0];
len=length(NodeLump.listdex);

for ik = 1:len
    str0=find(NT.list==NodeLump.list(ik));
    if isempty(str0)
        NT.list = [NT.list; NodeLump.list(ik)];     % avoid duplicate nodes
        NT.listdex = [NT.listdex; NodeLump.listdex(ik) + oft];
        NT.pos = [NT.pos; tmp];
        NT.num(1:2) = NT.num(1:2) + 1;
    end
end
NodeTower = NT;
oft = oft + len;  
end
