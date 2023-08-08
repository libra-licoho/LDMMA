function [y,time] = MOSEK_elastic_ac (settings,opts)
if ~isfield(opts,'lbd0'); opts.lbd0 = 0.01*ones(2,1); end
if ~isfield(opts,'r0'); opts.r0 = 12*ones(2,1); end
if ~isfield(opts,'epsilon'); opts.epsilon=1e-2; end
if ~isfield(opts,'miter');opts.miter = 10; end
if ~isfield(opts,'tol');opts.tol = 1e-4; end

lbd0 = opts.lbd0;  r0 = opts.r0;
epsilon = opts.epsilon; miter = opts.miter; tol = opts.tol;

ntr = settings.num_train; nval = settings.num_validate; nte = settings.num_test;
p = settings.num_features;

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
tic;

for iter = 1:miter
    clear prob;

    [r,res] = mosekopt('symbcon');

    coff_r = sqrt(lbd0./r0);
    coff_lbd = sqrt(r0./lbd0);

    prob.c = [zeros(3*p+ntr+5,1);1;zeros(p,1)];
    
    %linear constraints
    A1 = [sparse(p,p+4),speye(p),speye(p),X_tr',sparse(p,p+2)];
    A2u = [speye(p),sparse(p,2*p+ntr+6),-speye(p)];
    A2l = [speye(p),sparse(p,2*p+ntr+6),speye(p)];
    A2s = [sparse(1,p+1),1,sparse(1,2*p+ntr+4),ones(1,p)];
    A3u = [sparse(p,p+3),-ones(p,1),sparse(p,p),speye(p),sparse(p,p+ntr+2)];
    A3l = [sparse(p,p+3),ones(p,1),sparse(p,p),speye(p),sparse(p,p+ntr+2)];

    prob.a = [A1;A2u;A2l;A2s;A3u;A3l];
    prob.blc = [zeros(p,1);zeros(p,1);-inf*ones(p,1);0;zeros(p,1);-inf*ones(p,1)];
    prob.buc = [zeros(p,1);inf*ones(p,1);zeros(p,1);0;inf*ones(p,1);zeros(p,1)];
    prob.blx = [-inf*ones(p,1);zeros(4,1);-inf*ones(2*p+ntr,1);0;0;zeros(p,1)];
    prob.bux = [inf*ones(p,1);inf*ones(4,1);inf*ones(2*p+ntr,1);inf;inf;inf*ones(p,1)];

    %affine conic constraints
    prob.f = sparse([zeros(1,3*p+ntr+5),1,zeros(1,p); ...
                     X_val,zeros(nval,3*p+ntr+6); ...
                     zeros(1,p),1,zeros(1,3*p+ntr+5); ...
                     eye(p),zeros(p,3*p+ntr+6); ...
                     zeros(1,p),1,zeros(1,3*p+ntr+5); ...
                     zeros(1,p+2),-1,zeros(1,2*p+ntr+1),1,zeros(1,p+1); ...
                     zeros(p,p+4),eye(p),zeros(p,2*p+ntr+2); ...
                     zeros(1,p+2),1,zeros(1,2*p+ntr+1),1,zeros(1,p+1); ...
                     zeros(1,3*p+ntr+4),-2,zeros(1,p+1); ...
                     X_tr,zeros(ntr,3*p+ntr+6); ...
                     zeros(ntr,3*p+4),eye(ntr),zeros(ntr,p+2); ...
                     zeros(2,p),diag(coff_r),zeros(2,3*p+ntr+4); ...
                     zeros(2,p+2),diag(coff_lbd),zeros(2,3*p+ntr+2)]);
    prob.g = [0;-y_val;-1;zeros(p,1);1;0;zeros(p,1);0;epsilon+2*u;-y_tr;y_tr;zeros(4,1)];

    prob.cones = [res.symbcon.MSK_CT_QUAD nval+1 res.symbcon.MSK_CT_QUAD p+2 res.symbcon.MSK_CT_QUAD p+2 res.symbcon.MSK_CT_QUAD 2*ntr+5];
    
    [r,res] = mosekopt('minimize',prob);
    
    moseksol = res.sol.itr.xx'; % row vector
    
    % update and record
    r0 = max(tol,moseksol(p+1:p+2))';
    lbd0 = max(tol,moseksol(p+3:p+4))';
    s0 = moseksol(3*p+ntr+5);
    w0 = moseksol(3*p+5:3*p+4+ntr)';
    epsilon = epsilon/1.2;
   
    x_k = moseksol(1:p)';
    hist.x(iter,:) = x_k';
    hist.lbd(iter,:) = lbd0';
    hist.r(iter,:) = r0';
    hist.s(iter) = s0;
    hist.loss_val(iter) = 1/2*norm(X_val*x_k-y_val)^2;
    hist.loss_test(iter) = 1/2*norm(X_te*x_k-y_te)^2;
end
time = toc;
y = hist;
end