%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%C�digo Parcial1b.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Ciencia de los datos
% Introducci�n a la Ciencia de los Datos
% Universidad Tecnol�gica de Pereira - 2016 -2
% Julian David Echeverry, PhD
% Andr�s Marino �lvarez, PhD
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Cargar datos
clc, clear all, close all
load datosPrueba2
%% Normalizaci�n zscore variables de entrada datos de entranamiento
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
%% Evaluar datos de validaci�n
%Evalue el modelo entrenado sobre los datos de validaci�n aqui
Xtz = (Xtest -repmat(muz,size(Xtest,1),1))./repmat(sigz,size(Xtest,1),1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Coloque su c�digo aqui!
%%%%%%%%%%%%%%%%%%%%%
%guarde sus resultados
save('Prueba2Nombre.mat','letest')
