%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Decisión lineal - Fisher Discriminant Analysis
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Introducción a la Ciencia de los Datos
% Universidad Tecnológica de Pereira - 2016 -2
% Julian David Echeverry, PhD
% Andrés Marino Álvarez, PhD
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc;clear all;close all;format compact;

% Se crean dos clusters de datos de 100 puntos cada uno
N = 100;
X = 0.3*bsxfun(@plus, randn(N, 2), [6 6]);
Y = 0.6*bsxfun(@plus, randn(N, 2), [-2 -1]);

% Se crean las etiquetas para cada clase (1 para los datos de un cluster y 
% -1 para los demás)
b = [ones(N, 1); -ones(N, 1)];

% Se calculan dimensiones
[x_n, xdim]=size(X);
[y_n, ydim]=size(Y);

% Se calculan los vectores medios
muX=nanmean(X,1);
muY=nanmean(Y,1);

% Se sustrae la media a cada uno de los puntos
nX=X-repmat(muX,x_n,1);
nY=Y-repmat(muY,y_n,1);

nX(isnan(nX))=0;
nY(isnan(nY))=0;

% Se calculan las covarianzas intraclases
S1 = nX'*nX;  % Este es un estimado de la matriz de covarianza
S2 = nY'*nY;
Sw=S1+S2;

% Se calcula w
w=Sw\(muX-muY)';

% Se grafican los puntos del dataset
plot(X(:, 1), X(:, 2), 'bx'); hold on;
plot(Y(:, 1), Y(:, 2), 'rx'); 
xlim([-3 3]); ylim([-3 3]);

% Se grafica el vector resultante
plot(w(1),w(2),'gx')
% Se grafica una línea en la dirección del vector resultante
refline(w(2)/w(1),0)

% Se pueden verificar los valores de decisión
out1 = X*w;
out2 = Y*w;
