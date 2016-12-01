function [Cm,nnC] = A_norconfusionmatrix(labels,elabels)

nCi = unique(labels);
for ii = 1 : length(nCi)
    nnC(ii,1) = sum(labels == nCi(ii));
end

 Cm = confusionmat(labels,elabels);
 Cm = 100*Cm./repmat(nnC,1,size(Cm,2));
