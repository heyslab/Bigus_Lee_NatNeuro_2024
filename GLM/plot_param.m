function [param_plot] = ...
    plot_param(param_matrix,cell_mdl_matrix,selected_model_num,cellID,plotTitle, ylimt)


param_plot=cell(0);
selected_model_cellID = cell(1,3);

for k = 1:3  % ss,ls,sl
for i = 1:size(cell_mdl_matrix,1) %ismember(5, a)
    if ismember(cell_mdl_matrix(i,k),selected_model_num)
% cell_mdl_matrix(i,k) == selected_model_num
        selected_model_cellID{k} = [selected_model_cellID{k} cellID(i)];
    end
end
param_plot{k} = param_matrix{k}(selected_model_cellID{k},12:14);
end


figure

subplot(1,3,1)
boxchart(param_plot{1} )
xlabel('Position      Speed      Time')
pbaspect([ 1 2 1])
set(gca, 'XTick', [])
if ~isempty(ylimt)
ylim(ylimt)
end

subplot(1,3,2)
boxchart(param_plot{2} )
xlabel('Position      Speed      Time')
pbaspect([ 1 2 1])
set(gca, 'XTick', [])
if ~isempty(ylimt)
ylim(ylimt)
end
title(plotTitle)
subplot(1,3,3)
boxchart(param_plot{3} )
xlabel('Position      Speed      Time')
pbaspect([ 1 2 1])
set(gca, 'XTick', [])
if ~isempty(ylimt)
ylim(ylimt)
end

end