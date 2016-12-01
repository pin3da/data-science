function [L_SC,K,U,L] = A_sc_auto(X,Ng,nv,ndim)

%Inputs:
%X in\Real N x P: input matrix, P:features, N:samples

%Outputs:
%L_SC in \N N: label vector L_SC_I \in {1,2,...,Ng}
%K in \Real N x N: affinity matrix from Gaussian kernel-based graph building
%sigma in \Real^+: gaussian kernel bandwidth
%nv \in \N : number of nearest neighbors form building the graph
%representation from K
%K_ij = 0 if xi does not belong to the nv nearest neighbos of xj
%U\in Real N x Ng : Laplacian matrix eigenvectors
%val \in Real N : Laplacian eigenvalues
%data normalization


%X = zscore(X); %data normalization
if nargin < 3
    nv=2*ceil(sqrt(size(X,1)));
    ndim = Ng;
elseif nargin < 4
    ndim = Ng;
end

%euclidean distance computation
fprintf('Computing input distance...')
D = pdist(X);
Dc = squareform(D);
%building the affinity matrix
[~,ind]=sort(Dc,'ascend');
%Iv = ones(size(Dc));
for i=1:size(Dc,1)
    Dc(ind(nv+1:end,i),i) = 1e13;
    %Iv(ind(nv+1:end,i),i) = 0;
end
fprintf('done\n')
%kernel bandwidth estimation
vd = squareform(Dc-diag(diag(Dc)));
sigma = median(vd(vd<1e13));
clear vd Dc
%Gaussian kernel
K = exp(-squareform(D).^2/(2*sigma^2));
%number of nearest neighbors from N
fprintf('Building affinity matrix...')
%building the affinity matrix
[~,ind]=sort(K,'descend');
for i=1:size(K,1)
    K(ind(nv+1:end,i),i) = 0;
end
K = max(K,K');
%ensure positive definite
%Dkk = pdist(K);
%sigk = median(Dkk);
%K = exp(-squareform(Dkk).^2.^2/(2*sigk^2));
K = K*K'/(norm(K)^2);
%K = K.*K';

%nK = diag(K*K');
%K = (K*K')./(nK*nK');
fprintf('done\n')
%Computing unnormalized laplacian matrix
K = K-diag(diag(K)); % fixing the affinity matrix to ensure lower eigenvalues to be the most relevant ones
A = sum(K, 2);
%A = sqrt(1./A);
%A = spdiags(A, 0, n, n);
%L = eye(size(K)) - A * K * A;
%L = A * K * A;

%L = diag(A)\K;
if isempty(Ng)
    L = diag(A)-K;
    fprintf('Finding number of groups from a %d x %d Laplacian matrix',size(L,1),size(L,2))
    val = eig(L);
    val = sort(val,'ascend');
    [~,Ng] = max(diff(val));
    Ng = Ng -1;
end


fprintf('Number of groups %d - done\n',Ng)
fprintf('Doing spectral clustering...')
[L_SC,U,L,val] = sc(K,Ng,ndim);							%Grupos etiquetados
fprintf('done\n')

