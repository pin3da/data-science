function [kop,acop] = A_testsc_autoknn(X,labels,sigma,np,knn)
%we assume X is normalized

indices = A_crossval_bal(labels,np);
acc = zeros(np,numel(knn));
for i = 1 : np
    
    ltrain = labels(indices ~= i);
    ltest = labels(indices == i);
    Xtrain = X(indices ~= i,:);
    Xtest = X(indices == i,:);
    
    for kk = 1 : numel(knn)
        eltestc = A_testsc(Xtrain,ltrain,Xtest,sigma,'knn',knn(kk));
        acc(i,kk) =  100*sum(eltestc == ltest)/numel(ltest);
    end
    
end

macc = mean(acc);
[acop,ind] = max(macc);
kop = knn(ind);