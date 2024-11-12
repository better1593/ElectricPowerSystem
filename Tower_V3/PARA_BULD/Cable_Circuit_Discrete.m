function [Cable] = Cable_Circuit_Discrete(Cable, GLB)
%
%    Line.con = Seg.num =[ total# core# arm# seg# ]
%
dT = GLB.dT;
dZ = Cable.Seg.Lseg;                                        % length of seg

% (0) Output---------------------------------------------------------------

% (1) Cable input
Ht = Cable.Ht;
Cal.ord = Ht.ord;
Nord = Cal.ord;
Orng = 1:Nord;                                              %range of order
Npha = Cable.Line.con(2);                                   % # of cores
Prng=1:Npha;                                                %range of phase

Para = Cable.Para;
Cal.Tcov = Para.Tcov;
Cal.Tinv = Para.Tinv;

% (2) Calculate wave-equation parameters (Z and Y)
Rc = Para.Rc;                                               % Npha x Npha
Rca = Para.Rca;                                             % Npha x 1
Rac = Para.Rac;                                             % 1 x Npha
Ra = Para.Ra;                                               % 1 x 1

Lc = Para.Lc;                                               % Npha x Npha
Lca = Para.Lca;                                             % Npha x 1
Lac = Para.Lac;                                             % 1 x Npha
La = Para.La;                                               % 1 x 1

Cc = Para.Cc;                                               % Npha x Npha
Ca = Para.Ca;                                               % 1 x 1

% (3) VFIT parameters
Cal.Ec=zeros(Npha,Npha,Nord);                       % Bk
Cal.Bc=zeros(Npha,Npha,Nord);                       % Ak
for i = 1:Npha                                      % digonal elements
    Cal.Ec(i,i,:)=exp( Ht.c.a(i,i,Orng) * dT );     % Npha x Npha x Nord
    Cal.Bc(i,i,:)=Ht.c.r(i,i,Orng)./Ht.c.a(i,i,Orng).*(Cal.Ec(i,i,Orng)-1);   
end
Cal.Eca(1,:)=exp( Ht.ca.a(1,1,Orng) * dT );         % 1 x Npha x Nord  
Cal.Bca(1,:)=squeeze(Ht.ca.r(1,1,Orng)./Ht.ca.a(1,1,Orng)).'.*(Cal.Eca(1,Orng)-1); % !!!!
    
Cal.Eac(1,:)=exp( Ht.ac.a(1,1,Orng) * dT );         % 1 x Npha x Nord 
Cal.Bac(1,:)=squeeze(Ht.ac.r(1,1,Orng)./Ht.ac.a(1,1,Orng)).'.*(Cal.Eac(1,Orng)-1); % !!!!

Cal.Ea = exp( Ht.a.a * dT );                        % 1 x 1 x Nord 
Cal.Ba = Ht.a.r ./ Ht.a.a .* ( Cal.Ea -1 );         % 1 x 1 x Nord 

% (4a) Coef. of discrete core equations for Ic (Npha x Npha)
Cal.Dc = Rc/2 + Lc/dT + 0.5*sum(Cal.Bc,3);          % core equations
Cal.AIc = -inv(Cal.Dc)*(Rc/2 - Lc/dT);              % coef. of Ic term
Cal.AVc = -inv(Cal.Dc)/dZ;                          % coef. of Vc term

% (Npha x 1)
Cal.AIca1 = -inv(Cal.Dc) * (Rca + Lca/dT);          % Ic1a(k) term only
Cal.AIca2 = +inv(Cal.Dc) * Lca/dT;                  % Ic1a(k-1) term

% (4b) Coef. of discrete armor equations for Ia (1 x 1)-(1 X Npha)
Cal.Da = Ra/2 + La/dT + 1/2*sum(Cal.Ba,3);          % core equations  !!!!
Cal.AIa = -inv(Cal.Da)*(Ra/2 - La/dT);              % coef. of Ia term
Cal.AVa = -inv(Cal.Da)/dZ;                          % coef. of Va term
Cal.AIac1 = - inv(Cal.Da) * (Rac/2 + Lac/dT + sum(Cal.Bac,2)/2);%Ic1(k+0.5)
Cal.AIac2 = - inv(Cal.Da) * (Rac/2 - Lac/dT);       % Ic1(k-0.5)
Cal.ASRC = -inv(Cal.Da)/dZ;                         % 1 x 1 ind. voltage Vi

% （4c) Coef of discrete equatino for Vc and Va
Cal.BIc = -inv(Cc) * dT / dZ;                       % Npha x Npha
Cal.BIa = -inv(Ca) * dT / dZ;                       % 1 x 1
Cal.BSRC = -inv(Ca) * dT / dZ;                      % 1 x 1 dir stroke Is

% (4d) Coef of core discrete equation for hist. terem
for i = 1:Cal.ord                                   % Npha x Npha x Nord
    Cal.Phic(:,:,i) = -1/2*inv(Cal.Dc)*(Cal.Ec(Prng,Prng,i) + eye(Npha));
    Cal.Phica(1,i)  = -inv(Cal.Dc(1,1));            % 1 x Nord zero seq.
end

% (4e) Coef of armor discrete equation for hist. terem
for i = 1:Cal.ord                                   
    Cal.Phia(1,i)  = -1/2*inv(Cal.Da) * (Cal.Ea(1,i) + 1);  % 1 x Nord
    Cal.Phiac(1,i) = -1/2*inv(Cal.Da) * (Cal.Eac(1,i)+ 1);  % 1 x Nord  
end

% output
Cable.Cal = Cal;
end

