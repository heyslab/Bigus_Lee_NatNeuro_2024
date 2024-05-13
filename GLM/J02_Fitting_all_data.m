%% Load F file
clear all;
clc
[Ffile3,Ffilepath3]=uigetfile('*.mat','pick the F file','MultiSelect','on');


if iscell(Ffile3)==1
    for i=1:size(Ffile3,2)      
    %%%%%%%%%%% Settings %%%%%%%%%%%%%%%%
    Fit_mdl_and_save2(Ffile3{i},Ffilepath3)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    
    end
else

    Fit_mdl_and_save(Ffile3,Ffilepath3)
end