function [binned_data] = binData(input_data,binsize)


binned_data=zeros(floor(length(input_data)/binsize),1);


% Bin data
for i=1:floor(length(input_data)/binsize)
binned_data(i)= mean(input_data(round((i-1)*binsize+1):floor(round(i*binsize))));

end


end