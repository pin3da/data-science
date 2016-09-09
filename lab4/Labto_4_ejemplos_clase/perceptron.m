function [w] = perceptron(X,Y,w_init,eta)

w = w_init;
for iteration = 1 : 100             % Se puede optimizar poniendo un criterio de parada
  for ii = 1 : size(X,2)
    if sign(w'*X(:,ii)) ~= Y(ii)    % Dato mal clasificado?
      w = w + eta * X(:,ii) * Y(ii);% Sumar (o restar) a w
    end
  end
  sprintf('El porcentaje de mal clasificados es %2.2f',100*sum(sign(w'*X)~=Y)/size(X,2))
%   sum(sign(w'*X)~=Y)/size(X,2)      % Tasa de mal clasificados
end