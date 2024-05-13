function [posgrid, bins] = pos_map_JS(pos, nbins, trackLength)

% take the histogram
binSize =  trackLength/nbins;
bins = binSize/2:binSize: trackLength-binSize/2;

% store grid
posgrid = zeros(length(pos), nbins);

% loop over positions
for i = 1:length(pos)
    if isnan(pos(i))
        posgrid(i, :) = NaN(1,nbins);
    else
    % figure out the position index
    [~, idx] = min(abs(pos(i)-bins));
    posgrid(i, idx) = 1;
    end

end

end