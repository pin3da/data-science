%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%C�digo Parcial1a.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Ciencia de los datos
%Universidad Tecnol�gica de Pereira - 2016 -2
%Julian David Echeverry, PhD
%Andr�s Marino �lvarez, PhD
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Cargar datos
clc, clear all, close all
load datosPrueba1
%% Normalizaci�n zscore variables de entrada datos de entranamiento
[Xz,muz,sigz] = zscore(Xtrain);
sigz(sigz==0) = 1;
%% Entrenar modelo
% Entrene su modelo de estimaci�n aqui!!!:
%%%%%%%%%%%%%%%%%%%%%%%%%%
%yetrain = ?
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Estimar error de entrenamiento
close all
figure
plot(ytrain,'r','LineWidth',2)
hold on
plot(yetrain,'b')
legend({'Datos de entrenamiento','Estimaci�n'})
plot(yetrain,'b.')
xlabel('salida #','FontSize',14)
ylabel('y','FontSize',14)
title('Estimaci�n con minimos cuadrados')
figure
plot(100*abs(ytrain-yetrain)./abs(ytrain),'r','LineWidth',2)
xlabel('salida #','FontSize',14)
ylabel('error relativo [%]','FontSize',14)
ertrain = (100*norm(ytrain-yetrain)/norm(ytrain));
title(['Error Relativo ' num2str(ertrain,'%.2f') '[%]'])
showfigs_c(2)
%% Evaluar datos de validaci�n
%Evalue el modelo entrenado sobre los datos de validaci�n aqui
Xtz = (Xtest -repmat(muz,size(Xtest,1),1))./repmat(sigz,size(Xtest,1),1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Coloque su c�digo aqui!
%%%%%%%%%%%%%%%%%%%%%
%guarde sus resultados
save('Prueba1Nombre.mat','yetest')