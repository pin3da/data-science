function [cluster_labels U L la evd_time kmeans_time total_time] = sc(S, num_clusters,ndim)
%SC Spectral clustering using a sparse similarity matrix (t-nearest-neighbor).
%
%   Input  : A              : N-by-N sparse distance matrix, where
%                             N is the number of data
%            sigma          : sigma value used in computing similarity,
%                             if 0, apply self-tunning technique
%            num_clusters   : number of clusters
%
%   Output : cluster_labels : N-by-1 vector containing cluster labels
%            evd_time       : running time for eigendecomposition
%            kmeans_time    : running time for k-means
%            total_time     : total running time
%
%   Author : Wen-Yen Chen (wychen@alumni.cs.ucsb.edu)
%			 Chih-Jen Lin (cjlin@csie.ntu.edu.tw)

%
% Convert the sparse distance matrix to a sparse similarity matrix,
% where S = exp^(-(A^2 / 2*sigma^2)).
% Note: This step can be ignored if A is sparse similarity matrix.
%
% disp('Converting distance matrix to similarity matrix...');
tic;
n = size(S, 1);
% 
% if (sigma == 0) % Selftuning spectral clustering
%   % Find the count of nonzero for each column
%   disp('Selftuning spectral clustering...');
%   col_count = sum(A~=0, 1)';
%   col_sum = sum(A, 1)';
%   col_mean = col_sum ./ col_count;
%   [x y val] = find(A);
%   A = sparse(x, y, -val.*val./col_mean(x)./col_mean(y)./2);
%   clear col_count col_sum col_mean x y val;
% else % Fixed-sigma spectral clustering
%   disp('Fixed-sigma spectral clustering...');
%   A = A.*A;
%   A = -A/(2*sigma*sigma);
% end
% 
% % Do exp function sequentially because of memory limitation
% num = 2000;
% num_iter = ceil(n/num);
% S = sparse([]);
% for i = 1:num_iter
%   start_index = 1 + (i-1)*num;
%   end_index = min(i*num, n);
%   S1 = spfun(@exp, A(:,start_index:end_index)); % sparse exponential func
%   S = [S S1];
%   clear S1;
% end
% clear A;
% toc;

if nargin < 3
    ndim = num_clusters;
end

%
% Do laplacian, L = D^(-1/2) * S * D^(-1/2)
%
disp('Doing Laplacian...');
D = diag(sum(S, 2));% + (1e-10);
%D = sqrt(1./D); % D^(-1/2)
%D = spdiags(D, 0, n, n);
%L = eye(size(S)) - D * S * D;
%D = sqrt(1./D); % D^(-1/2)
%D = spdiags(D, 0, n, n);
L = D^(-.5) * S * D^(-.5);
%L = D - S; %find the largest eigenvalues
%clear D S;
time1 = toc;

%
% Do eigendecomposition, if L =
%   D^(-1/2) * S * D(-1/2)    : set 'LM' (Largest Magnitude), or
%   I - D^(-1/2) * S * D(-1/2): set 'SM' (Smallest Magnitude).
%
disp('Performing eigendecomposition...');
OPTS.disp = 0;
 %[V, val] = eigs(L, num_clusters, 'LM', OPTS);
[V, val] = eig(L);%, num_clusters, 'LM', OPTS);
la = diag(val);
[la,iii]=sort(la,'descend');
%[~,iii]=sort(la,'ascend');
V = V(:,iii(1:ndim));
time2 = toc;

%
% Do k-means
%
disp('Performing kmeans...');
% Normalize each row to be of unit length
%sq_sum = sqrt(sum(V.*V, 2)) + 1e-20;
%U = V ./ repmat(sq_sum, 1, num_clusters);
%clear sq_sum V;
U = V;
%cluster_labels = k_means(U, [], num_clusters);
% clc

[~, Ui] = kmeans(U, num_clusters,'emptyaction','singleton','start','cluster','replicates',2);
cluster_labels = kmeans(U, num_clusters,'emptyaction','singleton','start',Ui);
total_time = toc;

%
% Calculate and show time statistics
%
evd_time = time2 - time1;
kmeans_time = total_time - time2;
total_time;
disp('Finished sC!');
