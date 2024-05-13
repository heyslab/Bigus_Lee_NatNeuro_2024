function [testFit,trainFit,param,A,paraMat] = Fit2...
    (numFolds,posgrid, timegrid, lickgrid,...
    FiringRate,numcell)

numModels = 7;
for i = 1:numcell
A{1,i} = [timegrid{i} posgrid{i} lickgrid{i}]; 
A{2,i} = [timegrid{i} posgrid{i} ];
A{3,i} = [timegrid{i} lickgrid{i}];
A{4,i} = [posgrid{i} lickgrid{i}];
A{5,i} = timegrid{i};
A{6,i} = posgrid{i}; 
A{7,i} = lickgrid{i};
end



% Compute a filter, which will be used to smooth the firing rate
filter = gaussmf(-4:4,[8 0]); filter = filter/sum(filter);
dFF_smooth=[];
for j = 1:size(FiringRate,2)
dFF_smooth(:,j) = conv(FiringRate(:,j),filter,'same');
end


for i=1:numcell % Go over each cells     
    disp(i)
       for n = 1:numModels
        [testFit{i,n},trainFit{i,n},param{i,n},paraMat{i,n}]...
            = fit_model_gaussian_JS(A{n,i},dFF_smooth(:,i),numFolds);
       end
end

end