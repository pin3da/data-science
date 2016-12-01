function [logtrain, logtest] = A_crossval_balHO(label,pt)

%Balanced cross validation partition - kfold approach
%label: input labels
%pt:  percentage of training set

nC = unique(label);
logtrain = false(numel(label),1);
logtest = false(numel(label),1);
for i = 1 : numel(nC)
    indi = find(label==nC(i));
    [logtraini,logtesti] = crossvalind('HoldOut',numel(indi),1-pt);
    
    logtrain(indi)=logtraini;
    logtest(indi)=logtesti;
    
end

