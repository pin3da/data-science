%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Clustering - Expectation Maximization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%__________________________________________________________________________
% Copyright (C) 2016 Grupo de automatica - UTP
% Andres Marino Alvarez Meza
% Julian David Echeverry
% Ciencia de los datos
% $Id: test_sc_synthetic.m
%% Descripcion
% Aplicaci�n de algoritmos de clustering y extracci�n lineal de caracter�sticas
%% Required toolboxes
addpath(genpath('matSC'));
%% load your data
clc, close all, clear all
load data/DB
X = DB.sp2; %base de datos
figure
scatter(X(:,1),X(:,2),30,'filled')
xlabel('Caracteristica 1')
ylabel('Caracteristica 2')
title('datos de entrada')
%print('g4','-dpng')

Ng = 2; %numero de grupos

%% Agrupamiento herarquico
close all
D = squareform(pdist(X));
Link = linkage(D);
L_hc = cluster(Link,'maxclust',Ng);

figure 
dendrogram(Link);

figure
scatter(X(:,1),X(:,2),30,L_hc,'filled')
xlabel('Caracteristica 1')
ylabel('Caracteristica 2')
showfigs_c(2)

%% k means
close all
[L_km,ckme] = kmeans(X, Ng);

figure
scatter(X(:,1),X(:,2),30,L_km,'filled')
hold on
scatter(ckme(:,1),ckme(:,2),100,'y','filled')
hold off
xlabel('Caracteristica 1')
ylabel('Caracteristica 2')


%% spectral clustering
sig = median(squareform(D));
K = exp(-D.^2/(2*sig^2));

nv = 10;
[L_SC,K,U] = A_sc_auto(X,Ng,nv);

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
scatter(X(:,1),X(:,2),30,L_SC,'filled')
xlabel('Caracteristica 1')
ylabel('Caracteristica 2')

figure
scatter(U(:,1),U(:,2),30,L_SC,'filled')
xlabel('Base 1')
ylabel('Base 2')
title('Espacio de las bases de K')


showfigs_c(3)