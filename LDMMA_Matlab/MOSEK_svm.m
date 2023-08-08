function [y,time] = MOSEK_svm(settings,data,opts)
if ~isfield(opts,'lbd0'); opts.lbd0 = 0.1; end
if ~isfield(opts,'r0'); opts.r0 = 0.5; end
if ~isfield(opts,'epsilon'); opts.epsilon = 1e-2; end
if ~isfield(opts,'miter');opts.miter = 10; end

lbd0 = opts.lbd0;  r0 = opts.r0;
epsilon = opts.epsilon; miter = opts.miter;

ntr = settings.num_train; nval = settings.num_validate; nte = settings.num_test;
p = settings.num_features; CV = settings.num_CV; fold = settings.num_one_fold;

X_te = data.X_test; y_te = data.y_test;
X = data.X; y = data.y;

hist = struct();
hist.loss_val = zeros(miter,1);
hist.loss_test = zeros(miter,1);
hist.w = zeros(miter,p);
hist.c = zeros(miter,1);
hist.lbd = zeros(miter,1);
hist.r = zeros(miter,1);
hist.s = zeros(miter,1);

tic;

for iter = 1:miter
    clear prob;

    [r,res] = mosekopt('symbcon');

    coff_r = sqrt(lbd0/r0);
    coff_lbd = sqrt(r0/lbd0);

    prob.c = [sparse(CV*p+CV+p+3+(CV-1)*fold+p,1);ones(nval,1);sparse(ntr+CV*p+CV*2+p+2+3,1)];

    A1valp = cell(CV,1); A1val = [];
    A1trp = cell(CV,1); A1tr = [];
    for i = 1:CV
        auwval = [sparse(fold,(i-1)*p),y(data.iVal{i}).*X(data.iVal{i},:),sparse(fold,(CV-i)*p)];
        aucval = [sparse(fold,i-1),-y(data.iVal{i}),sparse(fold,CV-i)];
        aulbdv = [sparse(fold,(i-1)*fold),speye(fold),sparse(fold,(CV-i)*fold)];
        A1valp{i} = [auwval,aucval,sparse(fold,2*p+3+(CV-1)*fold),aulbdv,sparse(fold,ntr+CV*p+CV*2+p+2+3)];
        A1val = [A1val;A1valp{i}]; %loss_val
        auwtr = [sparse((CV-1)*fold,(i-1)*p),y(data.iTr{i}).*X(data.iTr{i},:),sparse((CV-1)*fold,(CV-i)*p)];
        auctr = [sparse((CV-1)*fold,i-1),-y(data.iTr{i}),sparse((CV-1)*fold,CV-i)];
        aulbdt = [sparse((CV-1)*fold,(i-1)*(CV-1)*fold),speye((CV-1)*fold),sparse((CV-1)*fold,(CV-i)*(CV-1)*fold)];
        A1trp{i} = [auwtr,auctr,sparse((CV-1)*fold,2*p+3+(CV-1)*fold+nval),aulbdt,sparse((CV-1)*fold,CV*p+CV*2+p+2+3)];
        A1tr = [A1tr;A1trp{i}]; %loss_tr
    end
    A2w = [-sqrt(2)*speye(CV*p),sparse(CV*p,CV+2*p+3+(CV-1)*fold+ntr+nval),speye(CV*p),sparse(CV*p,CV*2+p+2+3)];
    A2r = []; A2rp = cell(CV,1);
    for i = 1:CV
        aur = [sparse(2,(i-1)*2),speye(2),sparse(2,(CV-i)*2)];
        A2rp{i} = [sparse(2,CV*p+CV+p+2),-ones(2,1),sparse(2,(CV-1)*fold+p+nval+ntr+CV*p),aur,sparse(2,p+2+3)];
        A2r = [A2r;A2rp{i}];
    end
    ov = [-1;1];
    A3rho = [sparse(p,CV*p+CV),-sqrt(2)*speye(p),sparse(p,3+(CV-1)*fold+p+nval+ntr+CV*p+CV*2),speye(p),sparse(p,2+3)];
    A3sl = [sparse(2,CV*p+CV+p),-ones(2,1),ov,sparse(2,1+(CV-1)*fold+p+nval+ntr+CV*p+CV*2+p),speye(2),sparse(2,3)];

    A4p = cell(CV,1); A4 = [];
    for i = 1:CV
        A4p{i} = [sparse(p,CV*p+CV),-speye(p),sparse(p,3),X(data.iTr{i},:)'*diag(y(data.iTr{i})),sparse(p,p+nval+ntr+CV*p+CV*2+p+2+3)];
        A4 = [A4;A4p{i}];
    end
    A5 = [sparse(1,CV*p+CV+p+3),ones(1,(CV-1)*fold)*diag(y(data.iTr{i})),sparse(1,p+nval+ntr+CV*p+CV*2+p+2+3)];
    A6u = [speye(CV*p),sparse(CV*p,CV+p+3+(CV-1)*fold),repmat(-speye(p),[CV,1]),sparse(CV*p,nval+ntr+CV*p+CV*2+p+2+3)];
    A6l = [speye(CV*p),sparse(CV*p,CV+p+3+(CV-1)*fold),repmat(speye(p),[CV,1]),sparse(CV*p,nval+ntr+CV*p+CV*2+p+2+3)];
    A7 = [sparse(1,CV*p+CV+p),1,sparse(1,2),-ones(1,(CV-1)*fold),sparse(1,p),ones(1,nval),sparse(1,ntr+CV*p+CV*2+p+2),1,sparse(1,2)];
    A8 = [sparse(1,CV*p+CV+p+1),coff_lbd,sparse(1,(CV-1)*fold+p+nval+ntr+CV*p+CV*2+p+4),-1,sparse(1,1)];
    A9 = [sparse(1,CV*p+CV+p+2),coff_r,sparse(1,(CV-1)*fold+p+nval+ntr+CV*p+CV*2+p+4),-1];

    prob.a = [A1val;A1tr;A2w;A2r;A3rho;A3sl;A4;A5;A6u;A6l;A7;A8;A9];
    
    ob = [1;-1];
    prob.blc = [1*ones(nval+ntr,1);zeros(CV*p,1);repmat(ob,[CV,1]);zeros(p+2,1);zeros(CV*p+1,1);-inf*ones(CV*p,1);zeros(CV*p,1);epsilon;zeros(2,1)];
    prob.buc = [inf*ones(nval+ntr,1);zeros(CV*p,1);repmat(ob,[CV,1]);zeros(p+2,1);zeros(CV*p+1,1);zeros(CV*p,1);inf*ones(CV*p,1);epsilon;zeros(2,1)];

    prob.blx = [-inf*ones(CV*p+CV,1);-inf*ones(p,1);zeros(3,1);-inf*ones((CV-1)*fold,1);1e-6*ones(p,1);zeros(nval+ntr,1);-inf*ones(CV*p,1);repmat(ob,[CV,1]);-inf*ones(p,1);0;-inf;-inf*ones(3,1)];
    prob.bux = [inf*ones(CV*p+CV,1);inf*ones(p,1);inf*ones(3,1);inf*ones((CV-1)*fold,1);10*ones(p,1);zeros(nval+ntr,1);inf*ones(CV*p,1);inf*ones(CV*2,1);inf*ones(p,1);inf*ones(2,1);inf*ones(3,1)];

    prob.cones.type = [res.symbcon.MSK_CT_QUAD,res.symbcon.MSK_CT_QUAD,res.symbcon.MSK_CT_QUAD,res.symbcon.MSK_CT_QUAD,res.symbcon.MSK_CT_QUAD];

    sub = [];
    for i = 1:CV
        sub = [sub,CV*p+CV+p+3+(CV-1)*fold+p+nval+ntr+(i-1)*p+1:CV*p+CV+p+3+(CV-1)*fold+p+nval+ntr+i*p];
        sub = [sub,CV*p+CV+p+3+(CV-1)*fold+p+nval+ntr+CV*p+(i-1)*2+1,CV*p+CV+p+3+(CV-1)*fold+p+nval+ntr+CV*p+i*2];
    end
    sub = [sub,CV*p+CV+p+3+(CV-1)*fold+p+nval+ntr+CV*p+CV*2+1:CV*p+CV+p+3+(CV-1)*fold+p+nval+ntr+CV*p+CV*2+p+2];
    sub = [sub,CV*p+CV+p+3+(CV-1)*fold+p+nval+ntr+CV*p+CV*2+p+2+2,CV*p+CV+p+3+(CV-1)*fold+p+nval+ntr+CV*p+CV*2+p+2+3,CV*p+CV+p+3+(CV-1)*fold+p+nval+ntr+CV*p+CV*2+p+2+1];

    prob.cones.sub = sub;
    prob.cones.subptr = [1,p+3,2*p+5,3*p+7,4*p+9];

    [r,res] = mosekopt('minimize',prob);
    moseksol = res.sol.itr.xx';

    lbd0 = moseksol(CV*p+CV+p+2);
    r0 = moseksol(CV*p+CV+p+3);
    s0 = moseksol(CV*p+CV+p+1);
    w0 = [];
    for i = 1:CV
        w0 = [w0,moseksol((i-1)*p+1:i*p)'];
    end
    c0 = moseksol(CV*p+1:CV*p+CV)';
    w_bar0 = moseksol(CV*p+CV+p+3+(CV-1)*fold+1:CV*p+CV+p+3+(CV-1)*fold+p)';

    hist.lbd(iter) = lbd0;
    hist.r(iter) = r0;
    hist.s(iter) = s0;

    [loss_val,~] = svm_loss(X,y,CV,w0,c0,data,settings);
    hist.loss_val(iter) = loss_val;

    np = length(y);
    q = [1/2*lbd0*speye(p),sparse(p,1+np);sparse(1,p+1+np)];
    c = [sparse(p+1,1);1/np*ones(np,1)];

    a = [y.*X,-y,speye(np)];
    blc = [ones(np,1)];
    buc = [inf*ones(np,1)];
    blx = [-w_bar0;-inf;zeros(np,1)];
    bux = [w_bar0;inf;inf*ones(np,1)];
    
    [res_p] = mskqpopt(q,c,a,blc,buc,blx,bux);
    
    solution_p = res_p.sol.itr.xx';

end
time = toc;
y = hist;
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
