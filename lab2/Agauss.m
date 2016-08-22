function Phi = Agauss(X,mu,sig)
% Copyright (C) 2016 SPRG
% Andres Marino Alvarez Meza
% $Id: Agauss.m
%Estimación función base gaussiana (exponencial) con media mu y varianza
%sig

% Entradas:
%X: matriz de datos de entrada de N x P, N: datos, P: características
%u: matriz de medias referencia de Q x P
%sig: >=0 varianza función exponencial

% Salidas:
%Phi: matriz de representación no lineal de N x Q
if nargin < 3
    sig = median(squareform(pdist2(X,mu))); %varianza como la mediana de la matriz de distancias
end

D = pdist2(X,mu);

Phi = exp(-(D.^2)/(2*sig^2));
