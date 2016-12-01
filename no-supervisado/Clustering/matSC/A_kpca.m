function [Y,W,Kc,Valo,Vec] = A_kpca(K,d)
% Kernel principal component analysis method
% Y = K*W
% FORMAT [Y,W,Val,Vec] = A_kca(K,d)
% K     - input kernel matrix in R^{NxN}, N: # samples
% d     - subspace size, d>1 # dimensions, 0<d<1 amount of retained
% spectrum energy, d = 0 # of embedding dimensions = #eigenvalues > mean(eigenvalues)
% Y     - subspace matrix in R^{Nxd}
% W     - Projection matrix in R^{Pxd}
% Val   - eigenvalues vector (from K)
% Vec   - eigenvectors matrix, each column is an eigenvector (from K)
%__________________________________________________________________________
% Copyright (C) 2015 Signal Processing and Recognition Group
% Andres Marino Alvarez Meza
% $Id: A_kca.m 2015-10-29 $
n=size(K,1);
Mc = (eye(size(K))-(1/size(K,1))*ones(size(K,1),1)*ones(1,size(K,1)));
Kc = Mc*K*Mc;
%[Vec,Valo] = eigs(Kc,round(n/2));
[Vec,Valo] = eig(Kc);
Valo = diag(abs(real(Valo)));

%sorting eigenvalues
[Valo ival]=sort(Valo,'descend');
Vec = real(Vec(:,ival));
Valo = Valo(ival);

Val = Valo;
Val = Val./sum(Val);

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
for i = 1 : size(W,2)
   W(:,i)  = W(:,i)/sqrt(abs(Valo(i)));
end

Y = K*W;
