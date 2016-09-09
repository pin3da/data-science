%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Decisión lineal - Algoritmo del perceptron
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Introducción a la Ciencia de los Datos
% Universidad Tecnológica de Pereira - 2016 -2
% Julian David Echeverry, PhD
% Andrés Marino Álvarez, PhD
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc;clear all;close all;format compact;

% Datos de entrada
X1=[rand(1,100);rand(1,100);ones(1,100)];   % clase '+1'
X2=[rand(1,100);1+rand(1,100);ones(1,100)]; % clase '-1'
X=[X1,X2];

% Etiquetas de los datos [-1,+1];
Y=[-ones(1,100),ones(1,100)];

% Vector w inicial
w=[.5 .5 .5]';

% Se establece la tasa de aprendizaje
eta = 0.1;
% Se llama a la función que implementa el algoritmo de perceptron
wtag = perceptron(X,Y,w,eta);
% Basado en lo estimado, se predicen los datos
ytag = wtag'*X;

% Se grafican tanto los datos iniciales como las predicciones
figure;hold on
plot(X1(1,:),X1(2,:),'b.')
plot(X2(1,:),X2(2,:),'r.')

plot(X(1,ytag<0),X(2,ytag<0),'bo')
plot(X(1,ytag>0),X(2,ytag>0),'ro')
legend('clase -1','clase +1','pred -1','pred +1')