function Tower=Tower_Circuit_Source(Tower, GLB, LGT, index, Tfold)
% Generating LGT source data for Tower, Span and Cable
%   Soc.ISF (dir)
%   Soc.VSF (ind) for MNA equation
%   Soc.typ = 0/1/2 (no/dir/ind);
%   Soc.pos = [T, T_id, cir_id, phase_id, cond_id, node_id(Seg)]

% !!! No induced voltage source at Lump-element Tower (T-Type = 11)--------
if Tower.Info{1,2}==11
    return;                         
end
% -------------------------------------------------------------------------

ID = Tower.ID;
Info = Tower.Info;
Node = Tower.Node;

LSoc = LGT.Soc;
Soc.typ = LSoc.typ;
Soc.pos = LSoc.pos;
Soc.dat = [];
icur = LGT.Lch.curr;                            % stroke current

str=Tfold+"\Tower"+num2str(ID)+".mat";
% load('data.mat'); % include (a) data_SRC.direct (b) data_SRC.induced

if LSoc.typ==1
    if (LSoc.pos(1)~=1) || (LSoc.pos(2)~=ID)    % Dir LGT --> other
        Soc.pos=[];
        Tower.Soc = Soc;
        return;
    else
        nodename=Info{1,12};                    % node name of pole tip
        Ipos = find(Node.list==nodename);
        nodeid = Node.list(Ipos);               % node id
        Soc.pos(6) = nodeid;
        Soc.dat = icur;
    end
elseif LSoc.typ == 0
    if index==2
        dat=load(str);
        Soc.dat=dat.dat;
        Tower.Soc=Soc;
        return;
    end
  
    Pstr=Tower.WireP(:,1:3);
    Pend=Tower.WireP(:,4:6);
    Lne.x1=Pstr(:,1);   Lne.x2=Pend(:,1);
    Lne.y1=Pstr(:,2);   Lne.y2=Pend(:,2);
    Lne.z1=Pstr(:,3);   Lne.z2=Pend(:,3);
    Lne.tran.pt_start=[Lne.x1 Lne.y1 Lne.z1];
    Lne.tran.pt_end=[Lne.x2 Lne.y2 Lne.z2];
        
    delta=(Pend-Pstr);
    Lne.tran.L=sqrt(sum(delta.*delta,2));    % length of wire segments
        
    [Er_T, Ez_T] = E_Cal(LGT, Lne);
    
    GND = Tower.GND;
    [U_TL]= Cor_Lossy_ground(GLB, LGT, GND, Lne,Er_T,Ez_T);            
    dat=U_TL;                               % format=[(t1 t2..) (p1 p2..)]
    save(str,'dat');
    Soc.dat=dat;
end
% Output
Tower.Soc = Soc;   
end
