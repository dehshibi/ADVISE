function [y,t,optw,gs,C] = ADVISEkernel(x,tin,M)
% [y,t,optw,gs,C] = ADVISEkernel(x,t,W)
%
% Function 'ADVISEkernel' returns an optimized kernel density estimate
% using a Gauss kernel function with bandwidths locally adapted to data.
%
% Input arguments:
% x:    Sample data vector.
% tin (optinal): Points at which estimation are computed.
% W (optinal):
%       A vector of kernel bandwidths.
%       If W is provided, the optimal bandwidth is selected from the
%       elements of W.
%       * NOTE: Do not search bandwidths smaller than a sampling 
%       resolution of data.
%       If W is not provided, the program searches the optimal bandwidth
%       using a golden section search method.
%
% Output arguments:
% y:    Estimated density
% t:    Points at which estimation was computed.
%       The same as tin if tin is provided.
%       (If the sampling resolution of tin is smaller than the sampling
%       resolution of the data, x, the estimation was done at smaller
%       number of points than t. The results, t and y, are obtained by
%       interpolating the low resolution sampling points.)
% optw: Optimal kernel bandwidth.
% gs:   Stiffness constants of the variable bandwidth examined.
%       The stifness constant is defined as a ratio of the optimal fixed
%       bandwidth to a length of a local interval in which a fixed-kernel
%       bandwidth optimization was performed.
% C:    Cost functions of stiffness constants.
%
%-- Optimisation principle:
%       The optimization principle is based on minimising the estimated L2
%       loss function between the kernel estimate and an unknown underlying
%       density function. A simple assumption is that samples are drawn 
%       from the density independently of each other.
%
%   Iteratively computing optimal fixed-size bandwidths within local 
%   intervals provides the locally adaptive bandwidth. The optimal 
%   bandwidths are chosen so that they fall inside intervals that are 
%   $\gamma$ times larger than the optimal bandwidths themselves.
%   By minimising the L2 estimate, the parameter $\gamma$ is optimised.

%% Parameters Settings
% Number of bandwidths examined for optimisation.
if isempty(M)
    M = 80;            
end
% Window function
WinFunc = 'Gauss';
% Number of bootstrap samples
nbs = 1*1e2;        

try
    x = reshape(x,1,numel(x));
    
    if nargin == 1
        T = max(x) - min(x);
        [~,~,dt_samp] = find(sort(diff(sort(x))),1,'first');
        try
            arg1 = ceil(T/dt_samp);
        catch
            arg1 = 255;
        end
        tin = linspace(min(x),max(x), min(arg1,1e3));
        t = tin;
        x_ab = x( logical((x >= min(tin)) .*(x <= max(tin))) ) ;
    else
        T = max(tin) - min(tin);
        x_ab = x( logical((x >= min(tin)) .*(x <= max(tin))) ) ;
        [~,~,dt_samp] = find( sort(diff(sort(x_ab))),1,'first');
        
        if dt_samp > min(diff(tin))
            t = linspace(min(tin),max(tin), min(ceil(T/dt_samp),1e3));
        else
            t = tin;
        end
    end
    dt = min(diff(t));
    
    % Compute a globally optimal fixed bandwidth
    % Create a finest histogram
    y_hist = histc(x_ab,t-dt/2)/dt;
    L = length(y_hist);
    N = sum(y_hist*dt);
    
    %% Computing local MISEs and optimal bandwidths
    
    % Window sizes
    WIN = logexp(linspace(ilogexp(max(5*dt)),ilogexp(1*T),M));
    % Bandwidths
    W = WIN;
    
    c = zeros(M,L);
    for j = 1:M
        w = W(j);
        yh = fftkernel(y_hist,w/dt);
        % Computing local cost function
        c(j,:) = yh.^2 - 2*yh.*y_hist + 2/sqrt(2*pi)/w*y_hist;
    end
    
    
    optws = zeros(M,L);
    for i = 1:M
        Win = WIN(i);
        
        C_local = zeros(M,L);
        for j = 1:M
            % Computing local cost function
            C_local(j,:) = fftkernelWin(c(j,:),Win/dt,WinFunc);
        end
        
        [~,n] = min(C_local,[],1);
        optws(i,:) = W(n);
    end
    
    %% Search of the stiffness parameter
    % Search of the stiffness parameter of variable bandwidths.
    % Selecting a bandwidth w/W = g.
    
    
    % Initialization
    tol = 10^-5;
    a = 1e-12; b = 1;
    
    phi = (sqrt(5) + 1)/2;  % Golden ratio
    
    c1 = (phi-1)*a + (2-phi)*b;
    c2 = (2-phi)*a + (phi-1)*b;
    
    f1 = CostFunction(y_hist,N,t,dt,optws,WIN,WinFunc,c1);
    f2 = CostFunction(y_hist,N,t,dt,optws,WIN,WinFunc,c2);
    
    k = 1;
    while  ( abs(b-a) > tol*(abs(c1)+abs(c2)) ) && k < 30
        if f1 < f2
            b = c2;
            c2 = c1;
            c1 = (phi - 1)*a + (2 - phi)*b;
            
            f2 = f1;
            [f1,yv1,optwp1] = CostFunction(y_hist,N,t,dt,optws,WIN,WinFunc,c1);
            
            yopt = yv1 / sum(yv1*dt);
            optw = optwp1;
        else
            a = c1;
            c1 = c2;
            c2 = (2 - phi)*a + (phi - 1)*b;
            
            f1 = f2;
            [f2,yv2,optwp2] = CostFunction(y_hist,N,t,dt,optws,WIN,WinFunc,c2);
            
            yopt = yv2 / sum(yv2*dt);
            optw = optwp2;
        end
        
        gs(k) = (c1);
        C(k) = f1;
        k = k + 1;
    end
   
    %% Return results
    y = interp1(t,yopt,tin);
    optw = interp1(t,optw,tin);
    t = tin;
catch
    y = []; optw = []; t = [];
end


function [Cg, yv, optwp] = CostFunction(y_hist,N,t,dt,optws,WIN,WinFunc,g)
% Selecting w/W = g, bandwidth g=c1;

L = length(y_hist);
optwv = zeros(1,L);
for k = 1: L
    gs = optws(:,k)'./WIN;
    
    if g > max(gs)
        optwv(k) = min(WIN);
    else
        if g < min(gs)
            optwv(k) = max(WIN);
        else
            idx = find(gs >= g, 1, 'last');
            optwv(k) = g*WIN(idx);
        end
    end
end

% Nadaraya-Watson kernel regression
optwp = zeros(1,L);
for k = 1: L
    Z = feval(WinFunc,t(k)-t,optwv/g);
    optwp(k) = sum(optwv.*Z)/sum(Z);
end

% Density estimation with the variable bandwidth

% Baloon estimator (speed optimized)
idx = find(y_hist ~= 0);
y_hist_nz = y_hist(idx);
t_nz = t(idx);

yv = zeros(1,L);
for k = 1: L
    yv(k) = sum( y_hist_nz*dt.*Gauss(t(k)-t_nz,optwp(k)));
end

yv = yv *N/sum(yv*dt); % Rate

% Cost function of the estimated density
cg = yv.^2 - 2*yv.*y_hist + 2/sqrt(2*pi)./optwp.*y_hist;
Cg = sum(cg*dt);


function [y] = fftkernel(x,w)
L = length(x);
Lmax = L+3*w; % Takes 3 sigma to avoid aliasing

n = 2^(ceil(log2(Lmax)));

X = fft(x,n);

f = [-(0:n/2) (n/2-1:-1:1)]/n;

K = exp(-0.5*(w*2*pi*f).^2); % Gauss

y = ifft(X.*K,n);
y = y(1:L);

function [y] = fftkernelWin(x,w,WinFunc)
L = length(x);
Lmax = L+3*w; % Takes 3 sigma to avoid aliasing

n = 2^(ceil(log2(Lmax)));

X = fft(x,n);

f = [-(0:n/2) (n/2-1:-1:1)]/n;
t = 2*pi*f;

if strcmp(WinFunc,'Boxcar')
    % Boxcar
    a = sqrt(12)*w;
    K = 2*sin(a*t/2)./(a*t);
    K(1) = 1;
elseif strcmp(WinFunc,'Laplace')
    % Laplace
    K = 1 ./ ( 1+ (w*2*pi.*f).^2/2 );
elseif strcmp(WinFunc,'Cauchy')
    % Cauchy
    K = exp(-w*abs(2*pi*f));
else
    % Gauss
    K = exp(-0.5*(w*2*pi*f).^2);
end

y = ifft(X.*K,n);
y = y(1:L);

function y = Gauss(x,w)
y = 1/sqrt(2*pi)./w.*exp(-x.^2/2./w.^2);

function y = Laplace(x,w)
y = 1./sqrt(2)./w.*exp(-sqrt(2)./w.*abs(x));

function y = Cauchy(x,w)
y = 1./(pi*w.*(1+ (x./w).^2));

function y = Boxcar(x,w)
a = sqrt(12)*w;
y = 1./a; y(abs(x) > a/2) = 0; % Speed optimization


function y = logexp(x)
idx = x<1e2;
y(idx) = log(1+exp(x(idx)));

idx = x>=1e2;
y(idx) = x(idx);

function y = ilogexp(x)
idx = x<1e2;
y(idx) = log(exp(x(idx))-1);

idx = x>=1e2;
y(idx) = x(idx);