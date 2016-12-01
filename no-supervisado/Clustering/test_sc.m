%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Clustering - Expectation Maximization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%__________________________________________________________________________
% Copyright (C) 2016 Grupo de automatica - UTP
% Andres Marino Alvarez Meza
% Julian David Echeverry
% Inferencia Bayesiana
% $Id: test_sc.m
%% Descripcion
% Aplicación de algoritmos de clustering y extracción lineal de características
%% Required toolboxes
addpath(genpath('matSC'));
%%
clear all, close all
load data/coil
D = pdist(X);
D = squareform(D);

Ng = 20; %numero de grupos
%% PCA
[Y,W,Val] = A_pca(X,0.9);
close all
figure
scatter3(Y(:,1),Y(:,2),Y(:,3),30,labels,'filled'), colorbar
xlabel('Caracterisitca pca 1')
ylabel('Caracterisitca pca 2')
zlabel('Caracterisitca pca 2')
title('Espacio PCA')
%% Agrupamiento herarquico
close all
Dy = squareform(pdist(Y));
Link = linkage(Dy);
L_hc = cluster(Link,'maxclust',Ng);

figure 
dendrogram(Link);

figure
scatter(Y(:,1),Y(:,2),30,L_hc,'filled')
xlabel('Caracteristica 1')
ylabel('Caracteristica 2')
showfigs_c(2)

%% k means
close all
[L_km,ckme] = kmeans(Y, Ng);

figure
scatter(Y(:,1),Y(:,2),30,L_km,'filled')
hold on
scatter(ckme(:,1),ckme(:,2),100,'y','filled')
hold off
xlabel('Caracteristica 1')
ylabel('Caracteristica 2')


%% spectral clustering
sig = median(squareform(Dy));
K = exp(-Dy.^2/(2*sig^2));

nv = 30;
[L_SC,K,U] = A_sc_auto(Y,Ng,nv);

close all
figure
imagesc(D), colorbar
title('Distancia euclidea')
xlabel('muestra')
ylabel('muestra')

figure
imagesc(K), colorbar
title('Gaussiano ')
xlabel('muestra')
ylabel('muestra')

figure
scatter3(Y(:,1),Y(:,2),Y(:,3),30,L_SC,'filled')
xlabel('Caracteristica 1')
ylabel('Caracteristica 2')

figure
scatter3(U(:,1),U(:,2),U(:,3),30,L_SC,'filled')
xlabel('Base 1')
ylabel('Base 2')
title('Espacio de las bases de K')

%
figure
wi = [0.06, 0.06]; %tamaño de las imagenes en scatter
pi = 0.25; %porcentaje de imagenes a graficar [0,1]
idim = [128,128]; %tamaño de las imagenes 128*128 = 16384
plotit = false; %pintar imagen a imagen
graf2dimages(X,Y,L_SC,pi,wi,idim,plotit);

showfigs_c(2)


