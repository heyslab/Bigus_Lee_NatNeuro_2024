function [] = Fit_mdl_and_save(Ffile,Ffilepath)

fullFfile =[Ffilepath Ffile];
load(fullFfile);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n_pos_bins = 10;
n_time_bins = 10;
numFolds = 5;
trackLength = 112;
maxTime = 16.5;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Calculate matrix for behavior variables
for k = 1:3
    for i=1: size(FiringRate{k},1)
    FR{k}(:,i)=cell2mat(FiringRate{k}(i,:));

    posgrid{k}{i} = pos_map_JS(cell2mat(Position{k}(i,:)), n_pos_bins, trackLength);
    timegrid{k}{i} = pos_map_JS(cell2mat(Time{k}(i,:)), n_time_bins, maxTime);
    numsample = length(cell2mat(Licking{k}(i,:)));
    lickt = cell2mat(Licking{k}(i,:));
    lickgrid{k}{i} = zeros(numsample,2);
        for j = 1:numsample
           if lickt(j)==0
               lickgrid{k}{i}(j,1)=1;
           else
               lickgrid{k}{i}(j,2)=1;
           end
        end

    end
end

for k=1:3
[testFit{k},trainFit{k},param{k},Behavior_Variables{k},paraMat{k}] = Fit2...
    (numFolds,posgrid{k}, timegrid{k}, lickgrid{k},...
    FR{k}, size(FiringRate{k},1));
end

save([Ffilepath [Ffile '_1203.mat']]);
fprintf('File Saved');
end