function estlabelo = A_sortlabelsscK(U,labels,estlabel)
% sort labels according to reference, avoid matching problems
%from clustering random initialization - using similarities in RKHS basis

%find target centroids
ncen = unique(labels);
cen = zeros(numel(ncen),size(U,2));

for c = 1 : numel(ncen)
    %find corresponding label number in labels vector
    indc = labels == ncen(c);
    cen(c,:) = mean(U(indc,:));
end

nc = unique(estlabel);
estlabelo = zeros(numel(labels),1);
cene = zeros(nc,size(U,2));

for c = 1 : numel(nc)
    %find corresponding label number in labels vector
    ind = estlabel == nc(c);
    cene(c,:) = mean(U(ind,:));
    D = pdist2(cen,cene(c,:));
    [~,indm] = sort(D,'ascend');
    estlabelo(ind) = ncen(indm(1));
end
