function [y,time] = yalmip_elastic(settings,opts)
if ~isfield(opts,'lbd0'); opts.lbd0 = 0.01*ones(2,1); end
if ~isfield(opts,'r0'); opts.r0 = 12*ones(2,1); end
if ~isfield(opts,'epsilon'); opts.epsilon=1e-2; end
if ~isfield(opts,'miter');opts.miter = 5; end

lbd0 = opts.lbd0;  r0 = opts.r0;
epsilon = opts.epsilon; miter = opts.miter;

ntr = settings.num_train; nval = settings.num_validate; nte = settings.num_test;
p = settings.num_features;

% Xdata = data.X; ydata = data.y;
% X_tr = Xdata(1:ntr,:);
% X_val = Xdata(ntr+1:ntr+nval,:);
% X_te = Xdata(ntr+nval+1:ntr+nval+nte,:);
% y_tr = ydata(1:ntr);
% y_val = ydata(ntr+1:ntr+nval);
% y_te = ydata(ntr+nval+1:ntr+nval+nte);
[X_val,y_val] = generate_elastic_data(nval,p);
[X_tr,y_tr] = generate_elastic_data(ntr,p);
[X_te,y_te] = generate_elastic_data(nte,p);
   
u = 1/2*norm(y_tr)^2; 

hist = struct();
hist.loss_val = zeros(miter,1);
hist.loss_test = zeros(miter,1);
hist.x = zeros(miter,p);
hist.lbd = zeros(miter,2);
hist.r = zeros(miter,2);
hist.s = zeros(miter,1);
hist.solver_time = zeros(miter,1);

ops = sdpsettings('usex0',0,'solver','mosek','savedebug',1);

for iter=1:miter
   % epsilon = epsilon/4;
    
    tic;
    x = sdpvar(p,1); rho1 = sdpvar(p,1); rho2 = sdpvar(p,1); r = sdpvar(2,1);
    lbd = sdpvar(2,1); s = sdpvar(1,1); w = sdpvar(ntr,1); t = sdpvar(1,1);

    coff_r = lbd0./r0;
    coff_lbd = r0./lbd0;
    
    Constraints = [0.5/ntr*norm(X_tr*x-y_tr)^2+1/2*norm(w+y_tr)^2+1/2*(coff_r'*(r.^2)+coff_lbd'*(lbd.^2))+s<=epsilon+u;...
        1/2*(x'*x)-r(1) <= 0;norm(x,1)-r(2) <= 0;norm(rho2,inf)-lbd(2)<=0;norm([sqrt(2)*rho1;s-lbd(1)])-(s+lbd(1)) <= 0;...
        X_tr'*w+rho1+rho2 == 0; 0.5/nval*norm(X_val*x-y_val)^2-t <= 0];
    Object = t;
    
    ops.mosek.MSK_DPAR_INTPNT_CO_TOL_REL_GAP = 8.0e-4;
    yalmip_result = optimize(Constraints,Object,ops);
    hist.solver_time(iter)=yalmip_result.solvertime;

    lbd0 = value(lbd);
    r0 = value(r);
    x0 = value(x);  
    
    % record
    % hist.loss_val(iter) = 0.5/nval*norm(X_val*x0-y_val)^2;
    hist.loss_val(iter) = value(t);
    hist.x(iter,:) = x0';
    hist.lbd(iter,:) = lbd';
    hist.r(iter,:) = r';
    hist.s(iter) = s;
    hist.loss_test(iter) = 0.5/nte*norm(X_te*x0-y_te)^2;
end
time = toc;
y = hist; 
end