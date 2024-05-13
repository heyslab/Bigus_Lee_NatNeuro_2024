function [P_Map,P_sum] = Spatial_Map_per_Session(F,P,Num_Spatial_Bins)
% F is an array
% returning an array

spatial_bins = 0:600/Num_Spatial_Bins:600;

P_Map = zeros(1,Num_Spatial_Bins);
P_sum = zeros(1,Num_Spatial_Bins);

    for j = 1:Num_Spatial_Bins
      
        bininds = find(and(P > spatial_bins(j),P < spatial_bins(j+1)));
        P_Map(j) = nanmean(F(bininds));
        
%         with_spikes = find(F(bininds)~=0);
%         P_sum(j) = length(with_spikes)/length(bininds);
    if isnan( P_Map(j))
         P_Map(j)=0;
    end

    end
    P_Map = MovingAve(1,P_Map);

P_Map = P_Map';
end