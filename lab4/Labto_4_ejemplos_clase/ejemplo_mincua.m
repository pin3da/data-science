%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Decisión lineal - mínimos cuadrados
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Introducción a la Ciencia de los Datos
% Universidad Tecnológica de Pereira - 2016 -2
% Julian David Echeverry, PhD
% Andrés Marino Álvarez, PhD
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc;clear all;close all;format compact;

% Se crean dos clusters de datos de 100 puntos cada uno
N = 100;
X1 = 0.3*bsxfun(@plus, randn(N, 2), [6 6]);
X2 = 0.6*bsxfun(@plus, randn(N, 2), [-2 -1]);

% Se crea la matriz X de datos de entrada
X = [[X1; X2] ones(2*N, 1)];

% Se crean las etiquetas para cada clase (1 para los datos de un cluster y 
% -1 para los demás)
b = [ones(N, 1); -ones(N, 1)];

% Se resuelve el problema de mínimos cuadrados
z = lsqlin(X, b); % Empleando la función de MATLAB
w = pinv(X)*b;    % Empleando la operación vista en clase

% Se grafican los datos y se grafica la curva de decisión
y = -z(3)/z(2) - (z(1)/z(2))*X;
hold on;
plot(X(1:100, 1), X(1:100, 2), 'bx'); hold on;
plot(X(101:end, 1), X(101:end, 2), 'rx'); 
xlim([-3 3]); ylim([-3 3]);
plot(X, y, 'r');