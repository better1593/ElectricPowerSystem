function [Cable] = Cable_Circuit_Source(Cable, GLB, LGT, index, Tfold)  
%   Update Cablen.Soc.typ/dat/pos
%   Initilize Span.Soc.ISF/VSF
%   Pos: [x, y, 0, 0, 0) for IND
%        [T/S/0, ID, CirID, PhaseID, Seg(node)] for DIR
%   Cable.Soc.pos: 
%        [T/S, ID, CirID, Cond_ID, Seg(node)] for DIR

ID = Cable.ID;

LSoc = LGT.Soc;
Soc.typ=LSoc.typ;
Soc.pos=LSoc.pos;
Soc.dat=[];
icur = LGT.Lch.curr;        % lightning stroke current

Seg = Cable.Seg;
Nseg = Seg.Nseg;            % segment number

str=Tfold+"\Cable"+num2str(ID)+".mat";
% load('data.mat'); % include (a) data_SRC.direct (b) data_SRC.induced

if LSoc.typ==1
    if (LSoc.pos(1)~=3) || (LSoc.pos(2)~=ID)
        Soc.pos=[];
        Cable.Soc=Soc;
        return;
    else
        Soc.dat = icur;
    end
elseif LSoc.typ == 0
    if index==2
        dat=load(str);
        Soc.dat=dat.dat;
        Cable.Soc=Soc;
        return;
    end

    height = Cable.Line.rad(6);    
    Pstr = [Cable.Pole(1:2) height];
    Pend = [Cable.Pole(4:5) height];
    
    delta=(Pend-Pstr)/Nseg;
    x0=Pstr(1)+(0:Nseg)'*delta(1);
    y0=Pstr(2)+(0:Nseg)'*delta(2);
    z0=height+(0:Nseg)'*0;

    Lne.x1=x0(1:end-1,1);   Lne.x2=x0(2:end,1);
    Lne.y1=y0(1:end-1,1);   Lne.y2=y0(2:end,1);
    Lne.z1=z0(1:end-1,1);   Lne.z2=z0(2:end,1);
    
    Lne.tran.pt_start=[Lne.x1 Lne.y1 Lne.z1];
    Lne.tran.pt_end  =[Lne.x2 Lne.y2 Lne.z2];
    slen=sqrt(sum(delta.*delta,2));
    Lne.tran.L=repmat(slen,Nseg,1);
 
    [Er_T, Ez_T] = E_Cal(LGT, Lne);     

    GND = Cable.GND;  
    [U_TL]= Cor_Lossy_ground(GLB, LGT, GND, Lne,Er_T,Ez_T);   
    dat=U_TL;                               % format=[(t1 t2..) (p1 p2..)]
    save(str,'dat');
    Soc.dat=dat;
end
% Output
Cable.Soc=Soc;   
end