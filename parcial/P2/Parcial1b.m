%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Código Parcial1b.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Ciencia de los datos
% Introducción a la Ciencia de los Datos
% Universidad Tecnológica de Pereira - 2016 -2
% Julian David Echeverry, PhD
% Andrés Marino Álvarez, PhD
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Cargar datos
clc, clear all, close all
load datosPrueba2
%% Normalización zscore variables de entrada datos de entranamiento
[Xz,muz,sigz] = zscore(Xtrain);
sigz(sigz==0)=1;
%% Entrenar clasificador
%Coloque su codigo de entrenamiento aqui!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%letrain = ???
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%matriz de confusion
Ctrain = confusionmat(ltrain,letrain)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Evaluar datos de validación
%Evalue el modelo entrenado sobre los datos de validación aqui
Xtz = (Xtest -repmat(muz,size(Xtest,1),1))./repmat(sigz,size(Xtest,1),1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Coloque su código aqui!
%%%%%%%%%%%%%%%%%%%%%
%guarde sus resultados
save('Prueba2Nombre.mat','letest')
