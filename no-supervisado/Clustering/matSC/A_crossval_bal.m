function [indices] = A_crossval_bal(label,np)

%Balanced cross validation partition - kfold approach
%label: input labels
%np: number of partitions or percentage of training set

nC = unique(label);
indices = zeros(numel(label),1);
for i = 1 : numel(nC)
    indi = find(label==nC(i));
    [indic] = crossvalind('Kfold',length(indi),np);
    indices(indi)=indic;
    
end

