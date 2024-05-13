function [Shuffled_FR,Shuffled_b] = Shuffle_by_trial...
    (FR,behavior_varibales,number_of_trials)


trial_index = randperm(number_of_trials);
Shuffled_FR =[];
Shuffled_b =[];

for j=1:number_of_trials
    i = trial_index(j);
    Shuffled_FR =  [Shuffled_FR; FR((500*(i-1)+1):500*i,:)];
    Shuffled_b = [Shuffled_b; behavior_varibales((500*(i-1)+1):500*i,:)];
end

end

