function [y,time] = yalmip_svm(settings,data,opts)
if ~isfield(opts,'lbd0'); opts.lbd0 = 0.1; end
if ~isfield(opts,'r0'); opts.r0 = 0.5; end
if ~isfield(opts,'epsilon'); opts.epsilon = 0.1; end
if ~isfield(opts,'miter');opts.miter = 5; end

lbd0 = opts.lbd0;  r0 = opts.r0;
epsilon = opts.epsilon; miter = opts.miter; 

ntr = settings.num_train; nval = settings.num_validate; nte = settings.num_test;
p = settings.num_features; CV = settings.num_CV; fold = settings.num_one_fold;

X_te = data.X_test; y_te = data.y_test;
X = data.X; y = data.y;
multivector_val = 0;
multivector_tr = 0;

hist = struct();
hist.loss_val = zeros(miter,1);
hist.loss_test = zeros(miter,1);
hist.w = zeros(miter,p);
hist.c = zeros(miter,1);
hist.lbd = zeros(miter,1);
hist.r = zeros(miter,1);
hist.s = zeros(miter,1);

ops = sdpsettings('usex0',0,'solver','mosek','savedebug',1);

tic;
for iter = 1:miter
    w = sdpvar(p,CV); c = sdpvar(1,CV); rho = sdpvar(p,1); r = sdpvar(1,1);
    lbd = sdpvar(1,1); s = sdpvar(1,1); w_bar = sdpvar(p,1); v = sdpvar((CV-1)*fold,1);

    coff_r = lbd0/r0;
    coff_lbd = r0/lbd0;
%     for i = 1:CV
%         multivector_val = multivector_val + sum(max(1 - y(data.iVal{i},:).*(X(data.iVal{i},:)*w(:,i)-c(i)),0));
%     end
%     for i = 1:CV
%         multivector_tr = multivector_tr + sum(max(1 - y(data.iTr{i},:).*(X(data.iTr{i},:)*w(:,i)-c(i)),0));
%     end
    [multivector_val,multivector_tr] = svm_loss(X,y,CV,w,c,data,settings);

    Constraints = [multivector_tr+(coff_r*(r^2)+coff_lbd*(lbd^2))+s-sum(v) <= epsilon;...
        norm([sqrt(2)*rho;s-lbd])-(s+lbd) <= 0; w_bar >= 1e-6; w_bar <= 10];
    for i = 1:CV
        Constraints = [Constraints; 1/2*w(:,i)'*w(:,i)-r <= 0];
        Constraints = [Constraints; X(data.iTr{i},:)'*(diag(y(data.iTr{i}))*v) - rho == 0];
        Constraints = [Constraints; sum(diag(y(data.iTr{i}))*v) == 0];
        % Constraints = [Constraints; diag(y(data.iTr{i},:))*v == 0];
        Constraints = [Constraints; w(:,i) <= w_bar];
        Constraints = [Constraints; w(:,i) >= -w_bar];
    end
    for i = 1:(CV-1)*fold
        Constraints = [Constraints; v(i) >= 0; v(i) <= 1];
    end
    
    Object =  multivector_val;
    
    ops.mosek.MSK_DPAR_INTPNT_CO_TOL_REL_GAP = 8.0e-4;
    yalmip_result = optimize(Constraints,Object,ops);
%     hist.solver_time(iter)=yalmip_result.solvertime;

    lbd0 = value(lbd);
    r0 = value(r);
    w0 = value(w);  
    c0 = value(c);
    s0 = value(s);
    w_bar0 = value(w_bar);

    [loss_val,~] = svm_loss(X,y,CV,w0,c0,data,settings);
    hist.loss_val(iter) = loss_val;
    hist.lbd(iter) = lbd0;
    hist.r(iter) = r0;
    hist.s(iter) = s0;
    
    w1 = sdpvar(p,1); c1 = sdpvar(1,1);

    Constraints1 = [w1 <= w_bar0; w1 >= -w_bar0];
    Object1 = 1/length(y)*sum(max(1-y.*(X*w1-c1),0))+1/2*lbd0*norm(w1)^2;

    ops.mosek.MSK_DPAR_INTPNT_CO_TOL_REL_GAP = 8.0e-4;
    yalmip_result2 = optimize(Constraints1,Object1,ops);
    
    wt = value(w1);
    ct = value(c1);
    predict = sign(X_te*wt-ct);
    hist.w(iter,:) = wt';
    hist.c(iter) = ct;
    hist.loss_test(iter) = 0.5*mean(abs(predict-y_te));

end
y = hist;
time = toc;
end

function [loss_val,loss_tr] = svm_loss(X,y,CV,w,c,data,settings)
    loss_val = 0;
    loss_tr = 0;
    for i = 1:CV
        loss_val = loss_val + sum(max(1-y(data.iVal{i}).*(X(data.iVal{i},:)*w(:,i)-c(i)),0));
        loss_tr = loss_tr + sum(max(1-y(data.iTr{i}).*(X(data.iTr{i},:)*w(:,i)-c(i)),0));
    end
    loss_val = loss_val/settings.num_validate;
    loss_tr = loss_tr/settings.num_train;
end
