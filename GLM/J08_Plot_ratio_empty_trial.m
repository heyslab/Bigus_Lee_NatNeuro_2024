%%
timecell_sigmdl=sig_mdl_all;timecell_badmdl=badmdl;
%%
for session=1:10
for k=1:3 % go over trial type
good_stability{session,k}=[];
bad_stability{session,k}=[];
for i = 1:cell_count(session) % go over all cells
if ~isempty(timecell_sigmdl{session,k})    
    if ismember(i,timecell_sigmdl{session,k}(:,1))
    good_stability{session,k} = [good_stability{session,k}; ratio_empty_trial{session,k}(i)];
    end
end

if ~isempty(timecell_badmdl{session,k}) 
    if ismember(i,timecell_badmdl{session,k}(:,1))
    bad_stability{session,k} = [bad_stability{session,k}; ratio_empty_trial{session,k}(i)];
    end
end

end
end
end

%%
plot_good=[];plot_bad=[];
for session=1:10
for k=1:3 % go over trial type
plot_good =[plot_good; good_stability{session,k}];
plot_bad =[plot_bad; bad_stability{session,k}];
end
end
%%
k=3;
figure
% subplot(2,1,1)
histogram(plot_good,0:0.05:1,'Normalization','probability')
hold on
histogram(plot_bad,0:0.05:1,'Normalization','probability')
legend("good","other")
%%
[p,h]=ranksum(plot_good,plot_bad)
%%
tmatrix = NaN(size(plot_bad,1),2);
tmatrix(:,2)= plot_bad;
tmatrix(1:size(plot_good,1),1)=plot_good;

figure
boxchart(tmatrix)
%%
median(tmatrix,"omitnan")