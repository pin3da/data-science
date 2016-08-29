%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Modelo lineal y mínimos cuadrados
%Código demo_mincuadrados.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Ciencia de los datos
%Universidad Tecnológica de Pereira - 2016 -2
%Julian David Echeverry, PhD
%Andrés Marino Álvarez, PhD
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%1. Introducción al modelo lineal y minimos cuadrados
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1. función de costo nociones básicas
clc, close all, clear all
[X,Y] = meshgrid(-2:0.2:2,-2:0.2:2); %grilla de datos
Z = X.*exp(-X.^2 - Y.^2);%estimación de función en 3D
figure
surface(X,Y,Z)
xlabel('variable 1','FontSize',14)
ylabel('variable 2','FontSize',14)
zlabel('funcion de costo','FontSize',14)
grid on
view(3)

%% 2. optimizacion con restricciones
close all, clc, clear all
f = inline('x.^2','x'); %funcion costo
x= linspace(-4,4,200); %dominio variable x
g = inline('x+1','x'); %restriccion lineal
fx = feval(f,x); %evaluar funcion costo
gx = feval(g,x); %evaluar curva restricción
xo = -1;
figure
grid on
xlabel('x')
ylabel('f(x)')
plot(x,fx,'r','LineWidth',2)
hold on
plot(x,gx,'LineWidth',2)
plot(xo,feval(f,xo),'ro','MarkerFaceColor','r')
plot(0,0,'go','MarkerFaceColor','g');
xlabel('x','FontSize',14)
ylabel('f(x)','FontSize',14)
set(gca,'FontSize',14)
grid on
ylim([-5,5])
legend({'f(x)=x^2';'g(x)=x+1';['f(x)=' num2str(feval(f,xo)) ' (g(x)=0)'];'f(x)=0'},'Location','Best')

% Encuentre el mínimo de f(x) para la restricción g(x) = x^3-8 = 0
% analizando el gráfico de la función de costo y la restricción.

%% 3.  modelo lineal
clear all, clc, close all
x = linspace(0,1,500);
y = 2*x-1;
varn = 0.1;
noise = sqrt(varn)*randn(1,length(y));
yn = y + noise;
plot(x,yn,'.')
hold on
plot(x,y,'r','LineWidth',2)
legend({'Datos','Solución lineal'})
xlabel('x','FontSize',14)
ylabel('f(x)','FontSize',14)
set(gca,'FontSize',14)
grid on

% Cambien varn y explique que pasa con los datos

%% 4. solución modelo lineal por mínimos cuadrados - función sinc(t)
clc, close all, clear all
X = linspace(-6,6,200)';
y = sinc(X);
varn = 0.015;
noise = sqrt(varn)*randn(length(y),1);
yn = y + noise;
[ye,w] = linealmincua(X,yn);
disp(w)

plot(X,yn,'k.')
hold on
plot(X,y,'r','LineWidth',2)
plot(X,ye,'b','LineWidth',2)
legend({'Datos de entrada','Solución ideal','Solución obtenida'})
xlabel('X','FontSize',14)
ylabel('sinc(X)','FontSize',14)

% Cambie el dominio de X y verifique que pasa con el modelo lineal.

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 5. Mínimos cuadrados regularizados con representacion no lineal
clc, close all, clear all
X = linspace(-6,6,200)';
y = sinc(X);
varn = 0.025;
noise = sqrt(varn)*randn(length(y),1);
yn = y + noise;

sig = 0.2;
mu = linspace(-6,6,10)';
Phi = Agauss(X,mu,sig);

[ye0,w,C0] = linealmincua(Phi,yn);
[lambda,M] = lcurva(Phi,y);
ye = linealmincua(Phi,yn,lambda);

figure
plot(X,yn,'k.')
hold on
plot(X,y,'r','LineWidth',2)
plot(X,ye0,'c','LineWidth',2)
plot(X,ye,'b','LineWidth',2)
legend({'Datos de entrada','Solución ideal','Solución \lambda=0',['Solución \lambda=' num2str(lambda,'%.2f')]})
xlabel('X','FontSize',14)
ylabel('sinc(X)','FontSize',14)
title('Estimación sinc(X)')


figure
imagesc(C0), colorbar
xlabel('caracterisitica','FontSize',14)
ylabel('caracteristica','FontSize',14)
title('\Phi^T\Phi','FontSize',14)
figure
[~,Val0] = eig(C0);
stem(diag(Val0))
xlabel('j','FontSize',14)
ylabel('Valj','FontSize',14)
title('Valores propios matriz - \Phi^T\Phi','FontSize',14)


figure
[~,Val] = eig(C0+lambda*eye(size(C0)));
imagesc(C0+lambda*eye(size(C0))), colorbar
xlabel('caracterisitica','FontSize',14)
ylabel('caracteristica','FontSize',14)
title('\Phi^T\Phi+\lambda I','FontSize',14)
figure
stem(diag(Val))
xlabel('j','FontSize',14)
ylabel('Valj','FontSize',14)
title('Valores propios  - \Phi^T\Phi+\lambda I','FontSize',14)

showfigs_c(3)

% Cambie la varianza del ruido y la función de aproximación lineal
% (Sigmoidal, polinomial) y realice la aproximación de la función sinc.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 6. Practica con base de datos real (Kaggle)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

