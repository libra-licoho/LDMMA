function [y,time] = MOSEK_elastic(settings,data,opts)
if ~isfield(opts,'lbd0'); opts.lbd0 = 0.01*ones(2,1); end
if ~isfield(opts,'r0'); opts.r0 = 12*ones(2,1); end
if ~isfield(opts,'epsilon'); opts.epsilon=1e-2; end
if ~isfield(opts,'miter');opts.miter = 10; end
if ~isfield(opts,'tol');opts.tol = 1e-4; end

lbd0 = opts.lbd0;  r0 = opts.r0;
epsilon = opts.epsilon; miter = opts.miter; tol = opts.tol;

ntr = settings.num_train; nval = settings.num_validate; nte = settings.num_test;
p = settings.num_features;

% Xdata = csvread('/home/lucky_x/Documents/MATLAB/MMHSP_matlab/X.csv');
% ydata = csvread('/home/lucky_x/Documents/MATLAB/MMHSP_matlab/y.csv');
Xdata = data.X; ydata = data.y;
X_tr = Xdata(1:ntr,:);
X_val = Xdata(ntr+1:ntr+nval,:);
X_te = Xdata(ntr+nval+1:ntr+nval+nte,:);
y_tr = ydata(1:ntr);
y_val = ydata(ntr+1:ntr+nval);
y_te = ydata(ntr+nval+1:ntr+nval+nte);
%[X_val,y_val] = generate_elastic_data(nval,p);
%[X_tr,y_tr] = generate_elastic_data(ntr,p);
%[X_te,y_te] = generate_elastic_data(nte,p);
   
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
    
    prob.c = [sparse(3*p+5+ntr,1);1;sparse(nval+3*p+2*ntr+9,1)];

    A1val = [X_val,sparse(nval,4+2*p+ntr+2),-sqrt(2*nval)*speye(nval),sparse(nval,3*p+2*ntr+9)];
    A1tr = [X_tr,sparse(ntr,3*p+ntr+nval+6),-sqrt(2*ntr)*speye(ntr),sparse(ntr,ntr+9+2*p)];
    A2 = [sparse(ntr,3*p+4),-speye(ntr),sparse(ntr,nval+ntr+p+2),speye(ntr),sparse(ntr,2*p+9)];
    A3r = [sparse(2,p),diag(coff_r),sparse(2,3*p+nval+3*ntr+4),-speye(2),sparse(2,2*p+7)];
    A3lbd = [sparse(2,p+2),diag(coff_lbd),sparse(2,3*p+nval+3*ntr+4),-speye(2),sparse(2,2*p+5)];
    A4 = [sparse(1,3*p+ntr+4),2,sparse(1,nval+p+2*ntr+5),1,sparse(1,2*p+4)];
    A5u = [speye(p),sparse(p,2*p+ntr+nval+6),-speye(p),sparse(p,2*ntr+2*p+9)];
    A5l = [speye(p),sparse(p,2*p+ntr+nval+6),speye(p),sparse(p,2*ntr+2*p+9)];
    A6 = [sparse(1,p+1),-1,sparse(1,4+2*p+ntr+nval),ones(1,p),sparse(1,2*ntr+2*p+9)];
    A7u = [sparse(p,p+3),-ones(p,1),sparse(p,p),speye(p),sparse(p,3*p+3*ntr+nval+11)];
    A7l = [sparse(p,p+3),ones(p,1),sparse(p,p),speye(p),sparse(p,3*p+3*ntr+nval+11)];
    A8 = [sparse(p,p+4),speye(p),speye(p),X_tr',sparse(p,nval+2*ntr+3*p+11)];
    A9r = [sparse(2,p),-ones(2,1),sparse(2,3*p+3*ntr+nval+10),speye(2),sparse(2,2*p+2)];
    A9x = [-sqrt(2)*speye(p),sparse(p,3*p+3*ntr+nval+13),speye(p),sparse(p,p+2)];
    ov = [-1;1];
    A10sl = [sparse(2,p+2),ov,sparse(2,2*p+ntr+1),-ones(2,1),sparse(2,2*p+2*ntr+nval+8),speye(2),sparse(2,p)];
    A10rho = [sparse(p,p+4),-sqrt(2)*speye(p),sparse(p,3*p+3*ntr+nval+11),speye(p)];

    prob.a = [A1val;A1tr;A2;A3r;A3lbd;A4;A5u;A5l;A6;A7u;A7l;A8;A9r;A9x;A10sl;A10rho];

    prob.blc = [y_val;y_tr;y_tr;zeros(4,1);2*u+epsilon;-inf*ones(p,1);zeros(p,1);0;-inf*ones(p,1);zeros(p,1);zeros(p,1);1;-1;zeros(2*p+2,1);];
    prob.buc = [y_val;y_tr;y_tr;zeros(4,1);2*u+epsilon;zeros(p,1);inf*ones(p,1);0;zeros(p,1);inf*ones(p,1);zeros(p,1);1;-1;zeros(2*p+2,1)];

    prob.blx = [-inf*ones(p,1);zeros(4,1);-inf*ones(2*p+ntr,1);0;0;-inf*ones(nval,1);zeros(p,1);-inf*ones(2*ntr,1);zeros(5,1);1;-1;-inf*ones(p,1);0;0;-inf*ones(p,1)];
    prob.bux = [inf*ones(p,1);inf*ones(4,1);inf*ones(2*p+ntr,1);inf;inf;inf*ones(nval,1);inf*ones(p,1);inf*ones(2*ntr+5,1);inf*ones(p+2,1);inf*ones(p+2,1)]; 
    
    prob.cones.type = [res.symbcon.MSK_CT_QUAD,res.symbcon.MSK_CT_QUAD,res.symbcon.MSK_CT_QUAD,res.symbcon.MSK_CT_QUAD];
    
    sub = [3*p+ntr+6];
    for i = 1:nval
        sub = [sub,3*p+ntr+6+i];
    end %first cone

    for j = 1:p+2
        sub = [sub,5*p+3*ntr+nval+13+j];
    end %second cone

    for v = 1:p+2
        sub = [sub,4*p+3*ntr+nval+11+v]; 
    end %third cone
    sub = [sub,4*p+3*ntr+nval+11];
    for l = 1:2*ntr+4
        sub = [sub,4*p+ntr+nval+6+l];
    end %fourth cone
    prob.cones.sub = sub;
    prob.cones.subptr = [1,nval+2,nval+p+4,nval+2*p+6];

    
    [r,res] = mosekopt('minimize echo(0)',prob);
    
    moseksol = res.sol.itr.xx'; % row vector
    
    % update and record
    r0 = max(tol,moseksol(p+1:p+2))';
    lbd0 = max(tol,moseksol(p+3:p+4))';
    s0 = moseksol(3*p+ntr+5);
    w0 = moseksol(3*p+5:3*p+4+ntr)';
    epsilon = epsilon/4;
   
    x_k = moseksol(1:p)';
    hist.x(iter,:) = x_k';
    hist.lbd(iter,:) = lbd0';
    hist.r(iter,:) = r0';
    hist.s(iter) = s0;
    hist.loss_val(iter) = 0.5/nval*norm(X_val*x_k-y_val)^2;
    hist.loss_test(iter) = 0.5/nte*norm(X_te*x_k-y_te)^2;
end
time = toc;
y = hist;
end