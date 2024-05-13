function [Shuffle_F_Trace,Shuffle_V] = Shuffle_JH2(F_Temp,V)

M = bwlabel(F_Temp>0);
% this generates a vector M that relates the indicies 
% from F_Temp to the number of each sequential fluorescent trasient eg M=[000111000222200333

% % Calculate the mean length of ITI
% indsITI = [];
% for i=1:max(M)-1
% indsITI(i) = find(M==i+1,1,'first')-find(M==i,1,'last');
% end
% zerosthresh = mean(indsITI);


counter = 1;
MM = [1];

for u = 1:length(M)-1
% this generates a vector MM that use M and F_Temp(:,ii) 
% eg MM=[111222333444455666];
    if M(u)==M(u+1)
    MM(u+1)=counter;
    end

    if M(u)~=M(u+1)
    counter=counter+1;
    MM(u+1)=counter;
    end
end

P = MM;
Counter_Stop=1;
u=1;

% ZThreshDivisor=10;
while Counter_Stop==1    
ITI=length(find(MM==u));

% if it is a real ITI
%     if sum(F_Temp(find(MM==u)))==0 && ITI>zerosthresh/ZThreshDivisor
   if sum(F_Temp(find(MM==u)))==0 && ITI>2
    x=randi([1 ITI-2],1,1);
    MM(find(MM==u,1,'first')+x:length(MM)) = ...
        MM(find(MM==u,1,'first')+x:length(MM))+1;
    %P(find(MM==i,1,'first')+x:length(MM)) = P(find(MM==i,1,'first')+x:length(MM))+1;
    else
    u=u+1;
    end
        
    if u==max(MM)
    Counter_Stop=0;
    end
end
       
x = randperm(max(MM));
Shuffle_F_Trace = [];
% Shuffle_indx =[];
% indx = 1:length(F_Temp);

for u = 1:length(x)
Shuffle_F_Trace = [Shuffle_F_Trace F_Temp(find(MM==x(u)))'];
% Shuffle_indx = [Shuffle_indx indx(find(MM==x(u)))'];
end

if size(F_Temp,1)>0
    Shuffle_F_Trace=Shuffle_F_Trace';
end



cc = exist('V');
if cc ~= 0
% Shuffle_V = zeros(size(V));
Shuffle_V=[];
    for u = 1:length(x)
    Shuffle_V = [Shuffle_V; V(find(MM==x(u)),:)];
    end
else
    Shuffle_V =[];
end




end

