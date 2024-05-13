function [curve_smooth] = MovingAve(sl_window,curve)
% input: sl_window,curve

% Just to change the dimention if 
% input is not a matrix
if size(curve,1)==1 
    curve = curve';
end

curve_smooth = zeros(size(curve));
for i = 1:size(curve,2) % go over all cols
        for k = 1:size(curve,1) % go over all data points in col
            window = curve(max(1,k-sl_window):min(size(curve,1),k+sl_window),i); 
            % cut out the slideing window 
            % max and min are used to make sure margins are included
            curve_smooth(k,i) = mean(window);
        end
end

if size(curve_smooth,2)==1
    curve_smooth = curve_smooth';
end

end

