function [dFF_binned,bins ] = bin_dFF(dFF,binSize )

% take the histogram
nbins = ceil(max(dFF));
bins = binSize/2:binSize: nbins-binSize/2;

% store grid
dFF_binned = zeros(length(dFF),1);

% loop over positions
for i = 1:length(dFF)
   if dFF(i)==0
       dFF_binned(i)=0;
   else
    [~, idx] = min(abs(dFF(i)-bins));
     dFF_binned(i)=idx ;
   end
end

end
