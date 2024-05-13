% Load F file 
% For data that are just time cells
clear all;
clc

[Ffile,Ffilepath]=uigetfile('*.mat','pick the F file','MultiSelect','on');
load([Ffilepath Ffile]);

%% find time cells with bad model fit
for session = 1:10
for k = 1:3
    badmdl{session,k} =[];
    for cellID = 1:cell_count(session,k)
    if ~ismember(cellID,good_cell_all{session,k})
       badmdl{session,k} = ...
       [badmdl{session,k};[cellID, all_mdl_all{session,k}(cellID,:)]];   
    end
    end
end
end


%% Plot histogram for time cells
% selectedmdl=cell(0);
for i=1:10
for k=1:3
    if ~isempty(all_mdl_all{i,k})
        tt = all_mdl_all{i,k}(:,2);
        tt(all_mdl_all{i,k}(:,1)<0)=0;
        selectedmdl{i,k} = tt;
    else
    selectedmdl{i,k}=[];
    end
end
end

selectedmdl1=cell2mat(selectedmdl(:,1));
selectedmdl2=cell2mat(selectedmdl(:,2));
selectedmdl3=cell2mat(selectedmdl(:,3));
selectedmdl_all = [selectedmdl1;selectedmdl2;selectedmdl3];

figure
h=histogram(selectedmdl_all,-0.5:1:7.5,'FaceColor',"#D95319",'LineWidth',1)
set(findobj(gca,'type','line'),'linew',1)
set(gca,'FontSize',16)
set(gca,'XTick',[])
pbaspect([3 4 1])
% ylim([-0.2,0.2])
% title("bset model")


