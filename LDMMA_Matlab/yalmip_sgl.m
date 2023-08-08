function [y,time] = yalmip_sgl(settings,data,opts)
if ~isfield(opts,'lbd0'); opts.lbd0 = 0.1*ones(settings.num_group+1,1); end
if ~isfield(opts,'r0'); opts.r0 = 5*ones(settings.num_group+1,1); end
if ~isfield(opts,'epsilon'); opts.epsilon=1e-2; end
if ~isfield(opts,'miter');opts.miter = 15; end

lbd0 = opts.lbd0; r0 = opts.r0;
epsilon = opts.epsilon; miter = opts.miter;

ntr = settings.num_train; nval = settings.num_validate; nte = settings.num_test;
p = settings.num_features; m = settings.num_group; gs = p/m;

% Xdata = csvread('/home/lucky_x/Documents/MATLAB/MMHSP_matlab/X_sgl.csv');
% ydata = csvread('/home/lucky_x/Documents/MATLAB/MMHSP_matlab/y_sgl.csv');
Xdata = data.X; ydata = data.y;
X_tr = Xdata(1:ntr,:);
X_val = Xdata(ntr+1:ntr+nval,:);
X_te = Xdata(ntr+nval+1:ntr+nval+nte,:);
y_tr = ydata(1:ntr);
y_val = ydata(ntr+1:ntr+nval);
y_te = ydata(ntr+nval+1:ntr+nval+nte);
%[X_val,y_val] = generate_sgl_data(nval,p);
%[X_tr,y_tr] = generate_sgl_data(ntr,p);
%[X_te,y_te] = generate_sgl_data(nte,p);

u = norm(y_tr)^2; 

hist = struct();
hist.loss_val = zeros(miter,1);
hist.loss_test = zeros(miter,1);
hist.x = zeros(miter,p);
hist.solver_time = zeros(miter,1);
hist.lbd = zeros(miter,m+1);
hist.r = zeros(miter,m+1);

tic;
ops = sdpsettings('usex0',0,'solver','mosek');
for iter = 1:miter
%     epsilon = epsilon/4;
    coff_r = lbd0./r0;
    coff_lbd = r0./lbd0;
    
    x = sdpvar(p,1); rho1 = sdpvar(p,1); rho2 = sdpvar(p,1); r = sdpvar(m+1,1);
    lbd = sdpvar(m+1,1); w = sdpvar(ntr,1); 
    
    Object = norm(X_val*x-y_val);
%     Constraints = [norm(X_tr*x-y_tr)^2+coff_r'*(r.^2)+coff_lbd'*(lbd.^2)+norm(w+y_tr)^2<=2*epsilon+u;sqrt(ones(1,p/m)*reshape(x,p/m,m).^2)<=r(1:m)';...
%                 sqrt(ones(1,p/m)*reshape(rho1,p/m,m).^2)<=lbd(1:m)';norm(rho2,inf)<=lbd(m+1);norm(x,1)<=r(m+1);X_tr'*w+rho1+rho2==0];
    Constraints =  [norm(X_tr*x-y_tr)^2+coff_r'*(r.^2)+coff_lbd'*(lbd.^2)+norm(w+y_tr)^2<=epsilon+u;
        norm(rho2,inf)<=lbd(m+1);norm(x,1)<=r(m+1);X_tr'*w+rho1+rho2==0];
    for i = 1:m
        Constraints = [Constraints; norm(x((i-1)*gs+1:i*gs),2) <= r(i)];
    end
    for j = 1:m
        Constraints = [Constraints; norm(rho1((j-1)*gs+1:j*gs),2) <= lbd(j)];
    end

    ops.mosek.MSK_DPAR_INTPNT_CO_TOL_REL_GAP = 8.0e-4;
    yalmip_result = optimize(Constraints,Object,ops);

%     solver_time = yalmip_result.solvertime;
    lbd0 = value(lbd);
    r0 = value(r);
    x0 = value(x);
    
    hist.loss_val(iter) = 0.5/nval*norm(X_val*x0-y_val)^2;
    hist.x(iter,:) = x0';
    %hist.solver_time(iter) = yalmip_result.solvertime;
    hist.lbd(iter,:) = lbd0';
    hist.r(iter,:) = r';
    hist.loss_test(iter) = 0.5/nte*norm(X_te*x0-y_te)^2;
end
time = toc;
y = hist;
end