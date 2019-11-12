function  u = decomposition(m)
% u = decomposition(m)
% u: componente cartoon
% m: imagen de entrada
% v = m-u : componente de texturas
%
% a modo de simplicar la implementacion la funcion solo acepta imagenes de
% entrada de tamano 256x256. La adaptacion a otros tamanos es simple, basta
% con cambiar el tamano del filtro g y ajustar los parametros.
%
% Codigo basado en el paper:
% Image decomposition using adaptive second-order total generalized variation
% Jianlou Xu - Xiangchu Feng - Yan Hao - Yu Han
% DOI 10.1007/s11760-012-0420-3
%
% Mathias L.V.

    [x,y] = meshgrid(-127:128, -127:128);
    r = sqrt(x.^2 + y.^2);
    s = 1^2;
    g = 1./(1+0.00001*abs(gradiente(conv2(m, exp(-(abs(r).^2)/(2*s))/(2*pi*s), 'same'))).^2);


    alpha0 = 50;
    alpha1 = 50;
    beta = 0.2;
    mu = 0.01;
    
    n = 12;
    sigma = 1/sqrt(n);
    tau = 1/sqrt(n);
    

%     proj_p = @(x) x./max(sqrt(sum(x.^2, ndims(x)))./(alpha1*g), 1);
%     proj_q = @(x) x./max(sqrt(sum(x.^2, ndims(x)))/alpha0, 1);
    
    proj_p = @(x) x./max((abs(x))./(alpha1*g), 1);
    proj_q = @(x) x./max((abs(x))/alpha0, 1);
    
    z = zeros(size(m));
    d = zeros(size(m));
    
    % para computar u por PD
    u = m;
    u_barra = m;
    w = zeros([size(m), 2]);
    w_barra = zeros([size(m), 2]);
    p = zeros([size(m), 2]);
    q = zeros([size(m), 3]);
    
    for i = 1:100
        p_old = p;
        p = proj_p(p + sigma*(gradiente(u_barra)-w_barra));
        q = proj_q(q + sigma*e(w_barra));
        u_old = u;
        u = (tau*mu*(z+d) + u + tau*div(p))/(1+mu*tau);
        w_old = w;
        w = w + tau*(p_old + divh(q));
        u_barra = 2*u - u_old;
        w_barra = 2*w - w_old;
        
        rho1 = partial_x_mas(u - d);
        rho2 = partial_y_mas(u - d);
        z = (mu/(beta+4*mu)) * (fm1(z) + f_1(z) + cm1(z) + c_1(z) ...
            - f_1(rho1) + rho1 - c_1(rho2) + rho2) + (beta/(beta+4*mu))*m;
        
        d = d - u + z;
        
        update = (sum((u(:)-u_old(:)).^2)) / (sum((u_old(:)).^2));
        if update <= 3e-4 && i>=10
            update
            i
            break;
        end
    end
end

% Funciones auxiliares 
function x = partial_x_mas(m)
    x = zeros(size(m));
    x(1:end-1, :) = m(2:end, :) - m(1:end-1, :);
end

function x = partial_y_mas(m)
    x = zeros(size(m));
    x(:, 1:end-1) = m(:, 2:end) - m(:, 1:end-1);
end

function x = partial_x_menos(m)
    x = zeros(size(m));
    x(1, :) = m(1,:);
    x(2:end-1, :) = m(2:end-1,:) - m(1:end-2,:);
    x(end, :) = -m(end-1,:);
end

function x = partial_y_menos(m)
    x = zeros(size(m));
    x(:, 1) = m(:, 1);
    x(:, 2:end-1) = m(:, 2:end-1) - m(:, 1:end-2);
    x(:, end) = -m(:, end-1);
end

function x = gradiente(m)
    x = zeros([size(m), 2]);
    x(:,:,1) = partial_x_mas(m);
    x(:,:,2) = partial_y_mas(m);
end

function x = div(m)
    x = partial_x_menos(m(:,:,1)) + partial_y_menos(m(:,:,2));
end

function x = divh(m)
    x = zeros([size(m,1), size(m,2), 2]);
    x(:,:,1) = partial_x_menos(m(:,:,1)) + partial_y_menos(m(:,:,2));
    x(:,:,2) = partial_x_menos(m(:,:,2)) + partial_y_menos(m(:,:,3));
end

function x = e(m)
    x = zeros([size(m,1), size(m,2), 3]);
    x(:,:,1) = partial_x_mas(m(:,:,1));
    x(:,:,2) = 0.5 * (partial_x_mas(m(:,:,2)) + partial_y_mas(m(:,:,1)));
    x(:,:,3) = partial_y_mas(m(:,:,2));
end

function x = fm1(m)
    x = zeros(size(m));
    x(1:end-1, :) = m(2:end, :);
end

function x = f_1(m)
    x = zeros(size(m));
    x(2:end, :) = m(1:end-1, :);
end

function x = cm1(m)
    x = zeros(size(m));
    x(:, 1:end-1) = m(:, 2:end);
end

function x = c_1(m)
    x = zeros(size(m));
    x(:, 2:end) = m(:, 1:end-1);
end