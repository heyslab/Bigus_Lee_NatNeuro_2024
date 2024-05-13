%% function to find the right parameters given the model type
function [param_pos,param_time]...
    = find_param(param,modelType,numPos,numTime)

param_pos = []; param_time = [];

if all(modelType == [1 0]) 
    param_pos = param;
elseif all(modelType == [0 1]) 
    param_time = param;

elseif all(modelType == [1 1])
    param_pos = param(1:numPos);
    param_time = param(numPos+1:numPos+numTime);

end
    


