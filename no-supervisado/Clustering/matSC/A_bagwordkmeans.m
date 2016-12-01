function [Xm,labelm] = A_bagwordkmeans(Xdata,labels,nsc)
%bag of words representation based on kmeans clustering
%Xdata: N x P matrix
%nsc: number of words for each class

Nc = unique(labels);
Xm = [];
labelm = [];
for i = 1 : length(Nc)
    Nci(i) = sum(labels == Nc(i));
    indc = randperm(Nci(i));
    labelc = labels(labels == Nc(i));
    xc = Xdata(labels == Nc(i),:);
    if Nci(i) <= nsc
        Xm = [Xm;xc];
        labelm = [labelm;labelc];
    else
        %resampling with clustering
         %[clab,xcc] = k_means(xc, [], nsc);
         [~,xcc] = kmeans(xc, nsc,'emptyaction','singleton');
         [clab,xcc] = kmeans(xc, nsc,'emptyaction','singleton','start',xcc);
         Xm = [Xm;xcc];
        %Xm = [Xm;xc(indc(1:nsc),:)];
        labelm = [labelm;repmat(Nc(i),nsc,1)];%Este es el nuevo vector de etiquetas
    end
end