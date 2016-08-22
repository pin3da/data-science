function [ye,w,C] = linealmincua(X,y,lambda)
% Copyright (C) 2016 SPRG
% Andres Marino Alvarez Meza
% $Id: linealmincua.m

%Solución primal problema de estimación lineal univariada por mínimos cuadrados
%regularizados
% Entradas:
%X: matriz de datos de entrada de N x P, N: datos, P: características
%y: vector de datos de salida de N x 1
%lambda: >=0 parámetro de regularización.

% Salidas:
%ye: vector de datos estimados de salida de N x 1
%w: vector de proyección  de P x 1.
if nargin < 3
    lambda = 0;
end

C = X'*X;
w = (C+lambda*eye(size(X',1)))\X'*y;
ye = X*w;
