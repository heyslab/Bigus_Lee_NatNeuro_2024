function [IG_time,IG_pos,IG_licking] = allcell_Information_Gained(testFit)

[r2,correlation,log_llh,log_llh_diff,mse] = mdl_stats(testFit);

IG_time=[];
IG_pos=[];
IG_licking =[];

for k=1:3
for i = 1:size(testFit{k},1) % go over cells
IG_time(i,k) = log_llh{k}(i,2)-log_llh{k}(i,6);
IG_pos(i,k)= log_llh{k}(i,2)-log_llh{k}(i,5);
IG_licking(i,k) = 0;

% IG_time(i,k) = log_llh{k}(i,1)-log_llh{k}(i,4);
% IG_pos(i,k)= log_llh{k}(i,1)-log_llh{k}(i,3);
% IG_licking(i,k) = log_llh{k}(i,1)-log_llh{k}(i,2);

% elseif mdl_max_llh{k}(cells) == 2 % timegrid posgrid
% IG_time(cells,k) = log_llh{k}(i,2)-log_llh{k}(i,6);
% IG_pos(cells,k) = log_llh{k}(i,2)-log_llh{k}(i,5);
% IG_licking(cells,k) = NaN;
% 
% elseif mdl_max_llh{k}(cells) == 3 % timegrid lickgrid
% IG_time(cells,k) = log_llh{k}(i,3)-log_llh{k}(i,7);
% IG_pos(cells,k) = NaN;
% IG_licking(cells,k) = log_llh{k}(i,3)-log_llh{k}(i,5);
% 
% elseif mdl_max_llh{k}(cells) == 4 % posgrid lickgrid
% IG_time(cells,k) = NaN;
% IG_pos(cells,k) = log_llh{k}(i,4)-log_llh{k}(i,7);
% IG_licking(cells,k) = log_llh{k}(i,4)-log_llh{k}(i,6);
% 
% elseif mdl_max_llh{k}(cells) == 5 || mdl_max_llh{k}(cells) == 6|| mdl_max_llh{k}(cells) == 7 
% % single model
% IG_time(cells,k) = NaN;
% IG_pos(cells,k) = NaN;
% IG_licking(cells,k) = NaN;
% 
% else   keyboard
% end

end
end




end