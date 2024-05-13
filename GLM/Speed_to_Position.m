function [P] = Speed_to_Position(V,frame_rate)

time_interval=1/frame_rate; % seconds
dP=zeros(1,length(V));

for i=1:length(V)
    if V(i)>0
    dP(i) = V(i)*time_interval; % cm
    else 
     dP(i) = 0; % cm
    end
end

P=cumsum(dP); % cm


end