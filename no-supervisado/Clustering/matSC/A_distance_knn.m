function estlabel_test = A_distance_knn(Dtrain_test,Ytrain,k,est_mode)
% K-nearest neighbor classifier - euclidean distance
% kernel: labele_i = mode(labtrain(Omega)); Omega: set of K neighbors of
% x_i according to the kernel function in Km
% FORMAT [estlabel_test, estlabel_train] = A_kernel_knn(Km,labels,k,indices,i)
% Ktrain_test     - Ntrain x Ntest kernel matrix -> similarity between
% samples in RKHS - Ktrain_test = pdist
% Ytrain - Ntrain x 1 output vector
% k    - number of nearest neighbors
%__________________________________________________________________________
% Copyright (C) 2015 Signal Processing and Recognition Group
% Andres Marino Alvarez Meza
% $Id: A_kernel_knn.m 2015-10-30]$

%sorting the samples from kernel similarities in Km
[~,itrain_test] = sort(Dtrain_test,'ascend'); %training samples vs testing samples

%matrix of labels -> training and testing
LTest = repmat(Ytrain,1,size(Dtrain_test,2));
LTest = LTest(itrain_test);
if strcmp(est_mode,'mode')
    if k > 1
        estlabel_test = mode(LTest(1:k,:))';
    else
        estlabel_test = LTest(1,:)';
    end
else %mean based estimation
    if k > 1
        estlabel_test = mean(LTest(1:k,:))';
    else
        estlabel_test = LTest(1,:)';
    end
end