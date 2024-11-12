function PosNew = RotPos(Pos,theta)
% Perform rotation of coordinates (x y) of elements in Tower using Info(4)
% Pos = [x y]nx2, theta = angle (anti-clockwire)

rotM = [cosd(theta) -sind(theta); sind(theta) cosd(theta)]; % rot. matrix
PosNew = rotM*Pos'; 
PosNew = PosNew';
end
