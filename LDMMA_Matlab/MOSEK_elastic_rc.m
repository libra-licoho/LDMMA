function [y,time] = MOSEK_elastic_rc(settings,opts)
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

    coff_r = lbd0./r0;
    coff_lbd = r0./lbd0;
    
    prob.c = [sparse(3*p+5+ntr,1);1;sparse(nval+p+2*ntr+6,1)];

    A1val = [X_val,sparse(nval,4+2*p+ntr+2),-speye(nval),sparse(nval,p+2*ntr+6)];
    A1tr = [X_tr,sparse(ntr,3*p+ntr+nval+6),-speye(ntr),sparse(ntr,ntr+6)];
    A2 = [sparse(ntr,3*p+4),-speye(ntr),sparse(ntr,nval+ntr+p+2),speye(ntr),sparse(ntr,6)];
    A3r = [sparse(2,p),diag(coff_r),sparse(2,3*p+nval+3*ntr+4),-speye(2),sparse(2,4)];
    A3lbd = [sparse(2,p+2),diag(coff_lbd),sparse(2,3*p+nval+3*ntr+4),-speye(2),sparse(2,2)];
    A4 = [sparse(1,3*p+ntr+4),2,sparse(1,nval+p+2*ntr+5),1,sparse(1,1)];
    A5u = [speye(p),sparse(p,2*p+ntr+nval+6),-speye(p),sparse(p,2*ntr+6)];
    A5l = [speye(p),sparse(p,2*p+ntr+nval+6),speye(p),sparse(p,2*ntr+6)];
    A6 = [sparse(1,p+1),-1,sparse(1,4+2*p+ntr+nval),ones(1,p),sparse(1,2*ntr+6)];
    A7u = [sparse(p,p+3),-ones(p,1),sparse(p,p),speye(p),sparse(p,p+3*ntr+nval+8)];
    A7l = [sparse(p,p+3),ones(p,1),sparse(p,p),speye(p),sparse(p,p+3*ntr+nval+8)];
    A8 = [sparse(p,p+4),speye(p),speye(p),X_tr',sparse(p,nval+2*ntr+p+8)];

    prob.a = [A1val;A1tr;A2;A3r;A3lbd;A4;A5u;A5l;A6;A7u;A7l;A8];
%     A1 = [1/sqrt(nval)*data.X_validate,sparse(nval,4+2*p+ntr+2),-eye(nval),sparse(nval,p+2*ntr+7)];
%     A1_ = [1/sqrt(ntr)*data.X_train,sparse(ntr,3*p+ntr+nval+6),-eye(ntr),sparse(ntr,ntr+7)];
%     A2 = [sparse(p,p+4),eye(p),eye(p),data.X_train',sparse(p,nval+p+2*ntr+9)]; 
%     A3 = [sparse(ntr,3*p+4),eye(ntr),sparse(ntr,nval+p+ntr+2),-eye(ntr),sparse(ntr,7)];
%     A4 = [sparse(1,p),-1,sparse(1,5+2*p+ntr+nval),ones(1,p),sparse(1,2*ntr+7)];
%     A5 = [eye(p),sparse(p,2*p+6+ntr+nval),eye(p),sparse(p,2*ntr+7)];
%     A6 = [eye(p),sparse(p,2*p+6+ntr+nval),-eye(p),sparse(p,2*ntr+7)];
%     A7 = [sparse(p,p+2),ones(p,1),sparse(p,1),eye(p),sparse(p,2*p+3*ntr+nval+9)];
%     A8 = [sparse(p,p+2),-ones(p,1),sparse(p,1),eye(p),sparse(p,2*p+3*ntr+nval+9)];
% %     A9 = [sparse(1,p),sqrt(lbd0./r0)',sqrt(r0./lbd0)',sparse(1,2*p+ntr),1/u,sparse(1,1+nval+2*ntr+p),-1*ones(1,5),0,0];
%     A9_1 = [sparse(2,p),diag(sqrt(lbd0./r0)),sparse(2,3*p+3*ntr+nval+4),eye(2),sparse(2,5)];
%     A9_2 = [sparse(2,p),diag(sqrt(r0./lbd0)),sparse(2,3*p+3*ntr+nval+6),eye(2),sparse(2,3)];
%     A9_3 = [sparse(1,3*p+4+ntr),1/u,sparse(1,2*ntr+nval+p+5),-1,sparse(1,2)];
%     A10 = [sparse(1,4+3*p+ntr),1/u,sparse(1,6+nval+p+2*ntr),1,0];
    
%     prob.a = [A1;A1_;A2;A3;A4;A5;A6;A7;A8;A9_1;A9_2;A9_3;A10];

    prob.blc = [y_val;y_tr;y_tr;zeros(4,1);2*u+epsilon;-inf*ones(p,1);zeros(p,1);0;-inf*ones(p,1);zeros(p,1);zeros(p,1)];
    prob.buc = [y_val;y_tr;y_tr;zeros(4,1);2*u+epsilon;zeros(p,1);inf*ones(p,1);0;zeros(p,1);inf*ones(p,1);zeros(p,1)];
%     prob.blc = [1/sqrt(nval)*data.y_validate;1/sqrt(ntr)*data.y_train;sparse(p,1);data.y_train;0;sparse(p,1);-inf*ones(p,1);sparse(p,1);-inf*ones(p,1);sparse(5,1);u];
%     prob.buc = [1/sqrt(nval)*data.y_validate;1/sqrt(ntr)*data.y_train;sparse(p,1);data.y_train;0;inf*ones(p,1);sparse(p,1);inf*ones(p,1);sparse(p,1);sparse(5,1);u];

    prob.blx = [-inf*ones(p,1);zeros(4,1);-inf*ones(2*p+ntr,1);0;0;-inf*ones(nval,1);zeros(p,1);-inf*ones(2*ntr,1);zeros(5,1);1];
    prob.bux = [inf*ones(p,1);inf*ones(4,1);inf*ones(2*p+ntr,1);u+1/2*epsilon;inf;inf*ones(nval,1);inf*ones(p,1);inf*ones(2*ntr+5,1);1]; 
    
    prob.cones.type = [res.symbcon.MSK_CT_QUAD,res.symbcon.MSK_CT_RQUAD,res.symbcon.MSK_CT_RQUAD,res.symbcon.MSK_CT_QUAD];
    
    sub = [3*p+ntr+6];
    for i = 1:nval
        sub = [sub,3*p+ntr+6+i];
    end
    sub = [sub,3*p+ntr+5,p+3];
    for j = 1:p
        sub = [sub,p+4+j];
    end
        sub = [sub,4*p+3*ntr+nval+12,p+2];
    for v = 1:p
        sub = [sub,v]; 
    end
    sub = [sub,4*p+3*ntr+nval+11];
    for l = 1:2*ntr+4
        sub = [sub,4*p+ntr+nval+6+l];
    end
    prob.cones.sub = sub;
    prob.cones.subptr = [1,nval+2,nval+p+4,nval+2*p+6];
%     sub = [4*p+12+3*ntr+nval];
%     for i = 1:2*ntr+5
%         sub = [sub,4*p+6+ntr+nval+i];
%     end
%     sub = [sub,3*p+5+ntr,p+4];
%     for j = 1:p
%         sub = [sub,2*p+4+j];
%     end
%     sub = [sub,p+2,4*p+3*ntr+nval+13];
%     for k = 1:p
%         sub = [sub,k];
%     end
%     sub = [sub,3*p+ntr+6];
%     for l = 1:nval
%         sub = [sub,3*p+ntr+6+l];
%     end
%     prob.cones.sub = sub;
%     prob.cones.subptr = [1,2*ntr+7,2*ntr+p+9,2*ntr+2*p+11];
    
    [r,res] = mosekopt('minimize',prob);
    
    moseksol = res.sol.itr.xx'; % row vector
    
    % update and record
    r0 = max(tol,moseksol(p+1:p+2))';
    lbd0 = max(tol,moseksol(p+3:p+4))';
    s0 = moseksol(3*p+ntr+5);
    w0 = moseksol(3*p+5:3*p+4+ntr)';

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


