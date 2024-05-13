function [P_map] = CalculatePmap...
    (Fmatrix,Position,Num_Spatial_Bins,trackLength)


P_map=[];

for k=1:size(Fmatrix,2)
F=[];
F = Fmatrix(:,k)';
P_map(:,k) = Spatial_Map_per_Session(F,Position,Num_Spatial_Bins,trackLength);
end




end