function [MedianPlot] = ...
    plot_param_all(param_matrix,cell_mdl_matrix,plotTitle, ylimt)


param_plot=cell(0);

for k = 1:3  % ss,ls,sl
for i = 1:size(cell_mdl_matrix,1) 
if   cell_mdl_matrix(i,k)==0
    param_plot{k}(i) = nan;
else
param_plot{k}(i) = param_matrix{k}(i,cell_mdl_matrix(i,k));
end
end
end


figure
boxchart([param_plot{1}' param_plot{2}' param_plot{3}'])
xlabel('SS      LS     SL')
pbaspect([ 1 2 1])
set(gca, 'XTick', [])
if ~isempty(ylimt)
ylim(ylimt)
end
title(plotTitle)

MedianPlot = nanmedian([param_plot{1}' param_plot{2}' param_plot{3}']);
end