%graficar inmersion 2d con figuras en cada punto
%X in R {NxIr} matrix of N images, image resolution Ir, each row is a vec
%version of the image
%Y in R {Nx2} 2D embedding
%labels in Z {Nx1} label vector
%pi in [0,1] per of images to plot into 2D embedding
%wi image size into 2D plot, ex: wi =[0.4 0.4]; each image is shown into a
%box of size 0.4 x 0.4
%idim = [image width, image high]
%ni: numero de imagenes a graficar

function graf2dimages(X,Y,labels,pi,wi,idim,plotit)
%close all
%images size in 2D space
%w=0.04;
%h=0.04;
w = wi(1); h = wi(2);
scatter(Y(:,1),Y(:,2),30,labels,'filled');
xlabel('1-th coordinate')
ylabel('2-th coordinate')

set(gca,'ActivePositionProperty','Position') %eliminar labels para calculo de posicion
pos=get(gca,'Position');%obtener posiciones [left bottom width heigth]
xlimv=get(gca,'xlim'); %limites en x de grafico [xmin xmax]
ylimv=get(gca,'ylim'); %limites en y de grafico [ymin ymax]
%maximos y minimosObtene
Mx = xlimv(2);
mx = xlimv(1);
My = ylimv(2);
my = ylimv(1);
hold on;

c = numel(unique(labels));

for j = 1:c
    Xc = X(labels == j,:);
    Yc = Y(labels == j,:);
    n = size(Yc,1);
    r=round(n/round(pi*n));

    for i = 1:r:n
        %obtener left y bottom escalado para cada punto
        ppx = ((Yc(i,1)-mx)/(Mx-mx))*pos(3)+pos(1);
        ppy = ((Yc(i,2)-my)/(My-my))*pos(4)+pos(2);
        axes('position',[ppx ppy w h],'Xtick',[],'Ytick',[],'box','on');

        imagenP = reshape(Xc(i,:),idim(1),idim(2));
        Im(:,:,1) = imagenP';
        Im(:,:,2) = imagenP';
        Im(:,:,3) = imagenP';
        imshow(uint8(Im),[])
        %colormap(gray)
        set(gca,'Visible','off')
        if plotit
        drawnow
        end
    end

end
