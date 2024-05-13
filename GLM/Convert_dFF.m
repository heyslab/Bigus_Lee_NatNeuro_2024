function [spiketrain] = Convert_dFF(dFF)

bins = 0:1:ceil(max(dFF));
spiketrain = zeros(size(dFF));
    for i = 1:length(dFF)
    [~, idx] = min(abs(dFF(i)-bins));
    spiketrain(i)=idx-1;
    end

end