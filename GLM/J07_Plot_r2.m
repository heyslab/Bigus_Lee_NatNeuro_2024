% Load F file "run me"
% For data that are just time cells
clear all;
clc

[Ffile,Ffilepath]=uigetfile('*.mat','pick the F file','MultiSelect','on');
load([Ffilepath Ffile]);

%% Find the sigficant mdls and its r2 adjusted

for session =1:10
    for k=1:3
     if isempty(sig_mdl_all{session,k})
         r2best{session,k}=NaN(1,7);
     else
        best_mdl=sig_mdl_all{session,k}(:,5);
        if ~isnan(best_mdl)
            r2best_t=NaN(size(best_mdl,1),7);
            for cellnum = 1:size(best_mdl,1)
            r2best_t(cellnum,best_mdl(cellnum)) = r2_adj_all{session,k}(cellnum,best_mdl(cellnum));
            end
            r2best{session,k}=r2best_t;
            r2best_t=[];
            best_mdl=[];
        else
        r2best{session,k}=NaN(1,7);
        end
     end
    end
end


%% Plot r2 adj best
% reshape data
r2_best_new =[];
for k =1:3
r2_best_new =[r2_best_new; cell2mat(r2best(:,k))];
end

figure
b=boxchart(r2_best_new (:,5:7) );
b.JitterOutliers = 'on';
b.MarkerStyle = '.';
set(findobj(gca,'type','line'),'linew',2)
set(gca,'FontSize',16)
set(gca,'XTick',[])
pbaspect([3 4 1])
ylim([-0.1,0.25])
% title("r2 adjusted")
%%
median(r2_best_new,"omitnan")
sum(~isnan(r2_best_new))
%%
p=[];
for i=2:7
    [p(i)]=signrank(r2_best_new(:,i),0,'tail','right');
end
p
