function [ye,w,C] = linealmincua(X,y,lambda)
% Copyright (C) 2016 SPRG
% Andres Marino Alvarez Meza
% $Id: linealmincua.m

%Soluci�n primal problema de estimaci�n lineal univariada por m�nimos cuadrados
%regularizados
% Entradas:
%X: matriz de datos de entrada de N x P, N: datos, P: caracter�sticas
%y: vector de datos de salida de N x 1
%lambda: >=0 par�metro de regularizaci�n.

% Salidas:
%ye: vector de datos estimados de salida de N x 1
%w: vector de proyecci�n  de P x 1.
if nargin < 3
    lambda = 0;
end

C = X'*X;
w = (C+lambda*eye(size(X',1)))\X'*y;
ye = X*w;
