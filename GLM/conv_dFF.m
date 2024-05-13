function [dFF_conv] = conv_dFF(dFF,filter)

dFF_conv = conv(dFF,filter,'same');

% loop over positions
for i = 1:length(dFF)
   if dFF(i)==0
       dFF_conv(i)=0;
   end
end

end