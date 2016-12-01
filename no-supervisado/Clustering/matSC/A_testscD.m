function eltest = A_testscD(Xtrain,ltrain,Xtest,modee,knn)
% test label estimation from spectral clustering - distance-based estimation
% We assume that Xtrain and Xtest are normalized (zscore)

if nargin < 6
    knn = 0;
end

if strcmp(modee,'centroids') == 1 %centroids based estimation
    nc = unique(ltrain);
    cen = zeros(numel(nc),size(Xtrain,2));
    for i = 1 : numel(nc) %find centroids
        cen(i,:) = mean(Xtrain(ltrain == nc(i),:));
    end
    % find label of nearest centroid
    Dtrain_test = pdist2(cen,Xtest);
    eltest = A_distance_knn(Dtrain_test,nc,1,'mode');
    
elseif strcmp(modee,'knn')
    %kernel between training and testing set
    Dtrain_test = pdist2(Xtrain,Xtest);
    eltest = A_distance_knn(Dtrain_test,ltrain,knn,'mode');
end
