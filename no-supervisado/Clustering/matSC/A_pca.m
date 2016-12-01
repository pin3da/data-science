function [Y,W,Val,Vec] = A_pca(X,d)
% Automatic Principal component analysis method
% Y = X*W
% FORMAT [Y,W,Val,Vec] = A_pca(X,d)
% X     - input matrix in R^{NxP}, N: # samples, P: # features
% d     - subspace size, d>1 # dimensions, 0<d<1 amount of retained
% variance, d = 0 # of embedding dimensions = #eigenvalues > mean(eigenvalues)
% Y     - subspace matrix in R^{Nxd}
% W     - Projection matrix in in R^{Pxd}
% Val   - eigenvalues vector
% Vec   - eigenvectors matrix, each column is an eigenvector
%__________________________________________________________________________
% Copyright (C) 2015 Signal Processing and Recognition Group
% Andres Marino Alvarez Meza
% $Id: A_pca.m 2015-10-29 $
[n p]=size(X);
Xpp=X;
if n>p
    P=Xpp'*Xpp; %outter product pxp
else
    P=Xpp*Xpp'; %inner product nxn
end
[Vec Val]=eig(P);
Val = abs(diag(Val));
Val = Val./sum(Val);
%sorting eigenvalues
[Val ival]=sort(Val,'descend');
Vec = Vec(:,ival);

if d == 0 %eigenvalues > mean(eigenvalues)
    ivald = Val >= mean(Val);
    W = Vec(:,ivald);
elseif d > 0 && d < 1
    va = 0;
    W = [];
    i = 1;
    while va < d
        W = [W,Vec(:,i)];
        va = va+Val(i);
        i=i+1;
    end
elseif d >=  1 %# diemnsion = d
    W = Vec(:,1:d);
end
if n<p
    W = Xpp'*W*diag(Val(1:size(W,2)).^-.5);
end

for i = 1 : size(W,2)
    W(:,i) = W(:,i)/norm(W(:,i));
end
Y = Xpp*W;
