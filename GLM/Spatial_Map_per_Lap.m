function [Triggered_Ave,Number_Spikes_per_bin] = Spatial_Map_per_Lap(F,P,Num_Spatial_Bins)
% Returns nx60 mat

if size(P,1)>1
    P=P';
end
trackstart = min(P);
trackend = max(P);
binsize = (trackend-trackstart)/Num_Spatial_Bins;


Lap_i_Time_Start=[];
Lap_i_Time_End =[];
Lap_i_Time_Start(1)=1;

Diff_P = diff(P);
Diff_P = [Diff_P(1) Diff_P];
LLL_End_Lap = bwlabel(Diff_P<=-3);

for i=1:max(LLL_End_Lap)
    Lap_i_Time_End(i) = find(LLL_End_Lap==i,1,'first')-1;
    Lap_i_Time_Start(i+1) = find(LLL_End_Lap==i,1,'first');
end
Num_Laps = length(Lap_i_Time_End);


Triggered_Ave = zeros(Num_Laps,Num_Spatial_Bins);
Number_Spikes_per_bin = zeros(Num_Laps,Num_Spatial_Bins);

for k = 1:Num_Laps % discard the 1st lap (incomplete)
startposition = trackstart;
for j = 1:Num_Spatial_Bins
stopposition=startposition+binsize;
bininds = find(and(P > startposition,P < stopposition));
bininds_Lap = bininds(find(and(bininds>=Lap_i_Time_Start(k),bininds<=Lap_i_Time_End(k))));
% find data points in the k th lap that are within 
% the j th spatial bin
Triggered_Ave(k,j) = nanmean(F(bininds_Lap));
Number_Spikes_per_bin(k,j) = nnz(F(bininds_Lap));


startposition=stopposition;
end
Triggered_Ave(k,:) = MovingAve(1,Triggered_Ave(k,:));

end


end
