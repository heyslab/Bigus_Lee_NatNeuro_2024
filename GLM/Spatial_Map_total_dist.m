function [P_Map] = Spatial_Map_total_dist(F,P,Num_Spatial_Bins,binsize)
% F is an array
% returning an array

P_Map = zeros(Num_Spatial_Bins,1);
startposition = 0;

for j = 1:Num_Spatial_Bins
        stopposition = startposition+binsize; % add one spatial bin 
        bininds = find(and(P > startposition,P < stopposition));
        % find data points across all laps that are within 
        % the j th spatial bin
        P_Map(j) = mean(F(bininds));

        % take the corresponding F data points across all laps 
        % and take the mean 
        startposition = stopposition;
end
    P_Map = MovingAve(1,P_Map);
    
end