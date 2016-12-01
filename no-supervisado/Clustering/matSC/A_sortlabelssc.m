function estlabelo = A_sortlabelssc(labels,estlabel)
% sort labels according to reference, avoid matching problems
%from clustering random initialization

nc = unique(estlabel);
estlabelo = zeros(numel(labels),1);

for c = 1 : numel(nc)
    %find corresponding label number in labels vector
    ind = estlabel == nc(c);
    cc = mode(labels(ind));
    estlabelo(ind) = cc;
end
