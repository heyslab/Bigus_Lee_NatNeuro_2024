% Load F file "run me"
% for data that only have time cells
clear all;
clc

[Ffile2,Ffilepath2]=uigetfile('*.mat','pick the F file','MultiSelect','on');
load([Ffilepath2 Ffile2]);

%% Plot IG 
IG_matrix = [ reshape(IG_time_all,[],1) reshape(IG_pos_all,[],1)  reshape(IG_licking_all,[],1)];
IG_matrix(IG_matrix==0)= NaN;
% IG_matrix(IG_matrix<=0)= NaN;

figure
b=boxchart(IG_matrix );
b.BoxFaceColor = 	"#77AC30";
b.JitterOutliers = 'on';
b.MarkerStyle = '.';
ylim([-10,160])
set(findobj(gca,'type','line'),'linew',2)
set(gca,'FontSize',16)
set(gca,'XTick',[])
pbaspect([1 1 1])
%%
median(IG_matrix,"omitnan")
%%
[p,tbl,stats] = kruskalwallis(IG_matrix)
%%
[p,h]=ranksum(IG_matrix(:,1),IG_matrix(:,2) )
%%
[p,h]=ranksum(IG_matrix(:,1),IG_matrix(:,3) )
%%
[p,h]=ranksum(IG_matrix(:,2),IG_matrix(:,3) )

%% Plot IG by trial type 
IG_by_trial=cell(0);
figure
for k=1:3
IG_by_trial{k} = [ IG_time_all(:,k),IG_pos_all(:,k),IG_licking_all(:,k)];
IG_by_trial{k}(IG_by_trial{k} == 0)=NaN;
% IG_by_trial{k}(IG_by_trial{k} <= 0)=NaN;

subplot(1,3,k)
b=boxchart(IG_by_trial{k}  );
b.BoxFaceColor = 	"#77AC30";
b.JitterOutliers = 'on';
b.MarkerStyle = '.';

set(findobj(gca,'type','line'),'linew',2)
set(gca,'FontSize',16)
set(gca,'XTick',[])
pbaspect([1 1 1])
ylim([-10,180])
end
%%
k=1;
[p,h]=ranksum(IG_by_trial{k}(:,1) ,IG_by_trial{k}(:,2))
%%
[p,h]=ranksum(IG_by_trial{k}(:,1) ,IG_by_trial{k}(:,3))
%%
[p,h]=ranksum(IG_by_trial{k}(:,2) ,IG_by_trial{k}(:,3))

%% Count the number of significant models
sigcellnum=[];
for i=1:10
    for k=1:3
        sigcellnum(i,k) = size(sig_mdl_all{i,k},1);
    end
end

sum(sigcellnum)
sum(sum(sigcellnum))

%% Plot histogram for all significant models
selectedmdl=cell(0);
for k=1:3
    for i=1:10
        if isempty(sig_mdl_all{i,k})
        selectedmdl{i,k} = [];    
        else
    selectedmdl{i,k} = sig_mdl_all{i,k}(:,5);
        end
end
end

selectedmdl1=cell2mat(selectedmdl(:,1));
selectedmdl2=cell2mat(selectedmdl(:,2));
selectedmdl3=cell2mat(selectedmdl(:,3));
selectedmdl_all = [selectedmdl1;selectedmdl2;selectedmdl3];

figure
h=histogram(selectedmdl_all,0.5:1:7.5,'FaceColor',"#D95319",'LineWidth',1)
set(findobj(gca,'type','line'),'linew',1)
set(gca,'FontSize',16)
set(gca,'XTick',[])
pbaspect([3 4 1])
% ylim([-0.2,0.2])
% title("bset model")

%% Plot r2 adjusted for all significant cell for all models
r2_new =[];
for k =1:3
r2_new =[r2_new; cell2mat(r2_adj_all(:,k))];
end

figure
b=boxchart(r2_new );
b.JitterOutliers = 'on';
b.MarkerStyle = '.';
set(findobj(gca,'type','line'),'linew',2)
set(gca,'FontSize',16)
set(gca,'XTick',[])
pbaspect([2 1 1])
ylim([-0.2,0.3])
% title("r2 adjusted")

median(r2_new,"omitnan")

for i=1:7
    [p(i)]=ranksum(r2_new(:,i),0,'tail','right');
end
p

%% Plot llh diff for all significant cell for all models
log_llh_diff_new =[];
for k =1:3
log_llh_diff_new =[log_llh_diff_new; cell2mat(log_llh_diff_all(:,k))];
end

figure
b=boxchart(log_llh_diff_new );
b.BoxFaceColor = 	"#77AC30";
b.JitterOutliers = 'on';
b.MarkerStyle = '.';
ylim([-100,200])
% title("log likelihood increase")
set(findobj(gca,'type','line'),'linew',2)
set(gca,'FontSize',16)
set(gca,'XTick',[])
pbaspect([2 1 1])
% ylim([-0.2,0.2])
% title("r2")

median(log_llh_diff_new,"omitnan")

for i=1:7
    [p(i)]=ranksum(log_llh_diff_new(:,i),0,'tail','right');
end
p

%% Plot mse for all significant cell for all models
mse_new =[];
for k =1:3
mse_new =[mse_new; cell2mat(mse_all(:,k))];
end

figure
b=boxchart(mse_new );
b.JitterOutliers = 'on';
b.MarkerStyle = '.';
ylim([0,0.08])

median(mse_new,"omitnan")