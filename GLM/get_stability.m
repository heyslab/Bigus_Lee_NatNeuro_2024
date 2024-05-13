function [stability] = get_stability(FiringRate,sections)
edges = round(linspace(1,size(FiringRate,1)+1,sections+1));

for  cellnum  = 1:size(FiringRate,2)
     FR =  FiringRate(:,cellnum)';
     for  n = 1:sections
      FR_section(:,n) = FR(edges(n):(edges(n+1)-1));
      num_trials = floor(length(FR_section(:,n))/500);
      mean_map(:,n) = mean(reshape(FR_section(1:num_trials*500,n),[],500))';
     end

     rho = triu(corr(mean_map),1);
     stability(cellnum,1) = mean(rho(rho~=0),"omitnan");
     FR_section =[];rho=[];
end

end