function [eltest] = A_testsc(Xtrain,ltrain,Xtest,sigma,modee,knn)
% test label estimation from spectral clustering - similarity-based
% estimation in RKHS
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
    Ktrain_test = exp(-pdist2(cen,Xtest).^2/(2*sigma^2));
    eltest = A_kernel_knn(Ktrain_test,nc,1,'mode');
    
elseif strcmp(modee,'knn')
    %kernel between training and testing set
    Ktrain_test = exp(-pdist2(Xtrain,Xtest).^2/(2*sigma^2));
    eltest = A_kernel_knn(Ktrain_test,ltrain,knn,'mode');
end
