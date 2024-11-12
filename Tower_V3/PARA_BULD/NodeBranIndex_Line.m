function [Node,Bran,Meas,Cir,Seg]=NodeBranIndex_Line(data,Blok,Pole)
% Return Name and ID for Node and Bran
% Node.list/listdex/num
% Bran.list/listdex/num
% Meas.list/listdex
% With common node/bran and Nwir.nair/ngnd/nmea/comp
%           Seg.num = [tot# pha# sw# slen seg# depth]
% (0) Initialization
Seg = Blok.Seg;                 
Cir = Blok.Cir;                 
Lseg = Seg.Lseg;                % length of a segment
Nrow = size(data,1);
Ncon = Seg.num(1);              % total conductors of the line
Ngdw = Seg.num(3);              % total ground wire or armore

% (1)Read pole data, obtain Nseg
pos1 = Pole(1:3);
pos2 = Pole(4:6);
ds = pos2-pos1;
dis = sqrt(sum(ds.*ds,2));
Nseg =round(dis/Lseg);                
Seg.num(5) = Nseg;
Seg.num(6) = 0;                 % depth
Seg.Nseg = Nseg;

% (2) Read OHL/Cable data
STR = 'ABCN';
Nlist = [];
Blist = [];
Cir.dat = [];
for i = 1:Nrow
    STR0 = 'ABCN';
    str0 = string(data{i,1});       % type name
    ckid = data{i,3};               % circuit id
    npha = data{i,5};               % phase # 
    str2 = "CIR" + num2str(ckid,'%04d');
    switch str0
        case "CIRC"    % one cable only --> order: SWA + Core_A/B/C/N              
            if Ngdw == 1
                Blist = [Blist; "Y" + str2 + 'M'];
                Nlist = [Nlist; "X" + str2 + 'M'];      % Head common node
            end  
            for j = 1:npha
                Blist = [Blist; "Y" + str2 + STR0(j)];  % Bran
                Nlist = [Nlist; "X" + str2 + STR0(j)];  % Head common node
            end
            Seg.num(6) = data{i,11};                     % depth of cable                        
            Cir.dat = [Cir.dat; [ckid Ncon 0 0]];  % AM
        case "SW"       
            Blist = [Blist; "Y" + str2 + 'S'];                % Bran
            Nlist = [Nlist; "X" + str2 + 'S'];                % Head common node
            Cir.dat = [Cir.dat; [ckid 1 0 0]];          % SW
        case "CIRO"
            for j = 1:npha 
                Blist = [Blist; "Y" + str2 + STR0(j)];  % Bran
                Nlist = [Nlist; "X" + str2 + STR0(j)];  % Head common node
                Cir.dat = [Cir.dat; [ckid j 0 0]];      % SWend  
            end
        case "Meas"
% !!!!------------------------------------------------------------
            nmea = data{i,6};                       % # of measurment lines
            Meas.list = cell2mat(data(i:i+nmea-1,3:5));     % CK, pha, Seg
            Meas.flag = cell2mat(data(i:i+nmea-1,2));       % 1=I, 2=V
            Itmp = find(Meas.list(:,1)>5000);               % cable only
            if isempty(Itmp)                                % span
                Meas.listdex = [];                  % [cond_id seg_id]
                for ik = 1:size(Meas.list,1)
                    pos = [0 0 Meas.list(ik,1:2)];
                    cond_id= Cond_ID_Read(Cir,pos);
                    Meas.listdex = [Meas.listdex; [cond_id, Meas.list(ik,3)]];
                end
            else                                            % cable only
                Meas.listdex = Meas.list(:,2:3);    % [cond_id seg_id]
            end
            Meas.Ib = Meas.listdex(Meas.flag==1,:);
            Meas.Vn = Meas.listdex(Meas.flag==2,:);
% !!!!------------------------------------------------------------
   end
end

for j = 1:Nseg
    Node.list(:,j) = Nlist(:) + num2str(j,'%03d');
    Bran.list(:,j) = Blist(:) + num2str(j,'%03d');
    Node.listdex(:,j) = (1:Ncon)'+ (j-1)*Ncon;
    Bran.listdex(:,j) = (1:Ncon)'+ (j-1)*Ncon;
end
Node.list(:,Nseg+1) = Nlist(:) + num2str(Nseg+1,'%03d'); % Tail_node
Node.listdex(:,Nseg+1) = (1:Ncon)'+ Nseg*Ncon;          % Tail_node

Node.com = Node.list(:,1:Nseg:end);                 % Tail common node
Node.comdex = Node.listdex(:,1:Nseg:end);           % Tail common node

Nn = Ncon*(Nseg+1);
Nb = Ncon*Nseg;
Node.num = [Nn Nn 0 0];                             % wire parameters
Bran.num = [Nb 0 0 Nb 0 0];                         % lump parameters

end






