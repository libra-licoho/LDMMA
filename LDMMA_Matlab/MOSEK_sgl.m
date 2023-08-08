function [y,time] = MOSEK_sgl(settings,data,opts)
if ~isfield(opts,'lbd0'); opts.lbd0 = 5*ones(settings.num_group+1,1); end
if ~isfield(opts,'r0'); opts.r0 = 0.01*ones(settings.num_group+1,1); end
if ~isfield(opts,'epsilon'); opts.epsilon = 1e-4; end
if ~isfield(opts,'miter'); opts.miter = 10; end
if ~isfield(opts,'tol'); opts.tol = 1e-5; end

lbd0 = opts.lbd0; r0 = opts.r0;
epsilon = opts.epsilon; miter = opts.miter; tol = opts.tol;

ntr = settings.num_train; nval = settings.num_validate; nte = settings.num_test;
p = settings.num_features; m = settings.num_group; gs = p/m;

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

%record
hist = struct();
hist.x = zeros(miter,p);
hist.loss_val = zeros(miter,1);
hist.loss_test = zeros(miter,1);
hist.lbd = zeros(miter,m+1);
hist.r = zeros(miter,m+1);
tic;

for iter = 1:miter
    clear prob;

    [r,res] = mosekopt('symbcon');

    coff_r = sqrt(lbd0./r0);
    coff_lbd = sqrt(r0./lbd0);

    prob.c = [sparse(3*p+2*m+ntr+2,1);1;sparse(nval+2*ntr+p+2*m+3,1)];

    A1val = [X_val,sparse(nval,2*p+2*m+ntr+3),-speye(nval),sparse(nval,p+2*m+2*ntr+3)];
    A1tr = [X_tr,sparse(ntr,3*p+2*m+ntr+nval+3),-speye(ntr),sparse(ntr,ntr+2*m+3)];
    A2 = [sparse(ntr,3*p+2*m+2),-speye(ntr),sparse(ntr,ntr+p+nval+1),speye(ntr),sparse(ntr,2*m+3)];
    A3r = [sparse(m+1,p),diag(coff_r),sparse(m+1,3*p+3*ntr+nval+m+2),-speye(m+1),sparse(m+1,m+2)];
    A3lbd = [sparse(m+1,p+m+1),diag(coff_lbd),sparse(m+1,3*p+3*ntr+nval+m+2),-speye(m+1),sparse(m+1,1)];
    A4u = [speye(p),sparse(p,2*p+2*m+ntr+nval+3),-speye(p),sparse(p,2*ntr+2*m+3)];
    A4l = [speye(p),sparse(p,2*p+2*m+ntr+nval+3),speye(p),sparse(p,2*ntr+2*m+3)];
    A5 = [sparse(1,p+m),-1,sparse(1,m+2*p+ntr+nval+2),ones(1,p),sparse(1,2*ntr+2*m+3)];
    A6u = [sparse(p,p+2*m+1),-ones(p,1),sparse(p,p),speye(p),sparse(p,p+3*ntr+nval+2*m+4)];
    A6l = [sparse(p,p+2*m+1),ones(p,1),sparse(p,p),speye(p),sparse(p,p+3*ntr+nval+2*m+4)];
    A7 = [sparse(p,p+2*m+2),speye(p),speye(p),X_tr',sparse(p,nval+p+2*ntr+2*m+4)];

    prob.a = [A1val;A1tr;A2;A3r;A3lbd;A4u;A4l;A5;A6u;A6l;A7];

    prob.blc = [y_val;y_tr;y_tr;zeros(2*m+2,1);-inf*ones(p,1);zeros(p,1);0;-inf*ones(p,1);zeros(p,1);zeros(p,1)];
    prob.buc = [y_val;y_tr;y_tr;zeros(2*m+2,1);zeros(p,1);inf*ones(p,1);0;zeros(p,1);inf*ones(p,1);zeros(p,1)];

%     prob.blc = [y_val;y_tr;sparse(p,1);sparse(p,1);-inf*ones(p,1);0;sparse(p,1);-inf*ones(p,1);y_tr;sparse(m+1,1);sparse(m+1,1)];
%     prob.buc = [y_val;y_tr;sparse(p,1);inf*ones(p,1);sparse(p,1);0;inf*ones(p,1);sparse(p,1);y_tr;sparse(m+1,1);sparse(m+1,1)];

%     A2 = [sparse(p,p+2*m+2),eye(p),eye(p),X_tr',sparse(p,nval+p+2*ntr+2*m+5)];
% 
%     A3 = [eye(p),sparse(p,2*m+2*p+ntr+nval+3),eye(p),sparse(p,2*ntr+2*m+4)];
%     A4 = [eye(p),sparse(p,2*m+2*p+ntr+nval+3),-eye(p),sparse(p,2*ntr+2*m+4)];
%     A5 = [sparse(1,p+m),-1,sparse(1,2*p+m+ntr+nval+2),ones(1,p),sparse(1,2*ntr+2*m+4)];
%     A6 = [sparse(p,p+2*m+1),ones(p,1),sparse(p,p),eye(p),sparse(p,p+3*ntr+nval+2*m+5)];
%     A7 = [sparse(p,p+2*m+1),-ones(p,1),sparse(p,p),eye(p),sparse(p,p+3*ntr+nval+2*m+5)];
%     A9_1 = [sparse(m+1,p),diag(sqrt(lbd0./r0)),sparse(m+1,m+3*p+3*ntr+nval+2),eye(m+1),sparse(m+1,m+3)];
%     A9_2 = [sparse(m+1,p+m+1),diag(sqrt(r0./lbd0)),sparse(m+1,3*p+3*ntr+nval+m+2),eye(m+1),sparse(m+1,2)];
% 
%     prob.a = [A1val;A1tr;A2;A3;A4;A5;A6;A7;A8;A9_1;A9_2];
    prob.blx = [-inf*ones(p,1);zeros(2*m+2,1);-inf*ones(2*p+ntr,1);0;-inf*ones(nval,1);zeros(p,1);-inf*ones(2*ntr,1);sparse(2*m+2,1);u+epsilon];
    prob.bux = [inf*ones(p,1);inf*ones(2*m+2,1);inf*ones(2*p+ntr,1);inf;inf*ones(nval,1);inf*ones(p,1);inf*ones(2*ntr,1);inf*ones(2*m+2,1);u+epsilon];

    prob.cones.type = [res.symbcon.MSK_CT_QUAD,res.symbcon.MSK_CT_QUAD];
    for ind = 1:m
        prob.cones.type = [prob.cones.type,res.symbcon.MSK_CT_QUAD,res.symbcon.MSK_CT_QUAD];
    end

    sub = [];
    for i = 1:nval+1
        sub = [sub,3*p+ntr+2*m+2+i];
    end
    sub = [sub,4*p+4*m+3*ntr+nval+6];
    for j = 1:2*ntr+2*m+2
        sub = [sub,4*p+2*m+3+ntr+nval+j];
    end
    for i = 1:m
        sub = [sub,p+i];
        for j = 1:gs
            sub = [sub,(i-1)*gs+j];
        end
    end
    for i = 1:m
        sub = [sub,p+m+1+i];
        for j = 1:gs
            sub = [sub,p+2*m+2+(i-1)*gs+j];
        end
    end

    subptr = [1,nval+2];
    for i = 1:2*m
        subptr = [subptr,nval+2*ntr+2*m+4+(i-1)*(gs+1)+1];
    end
    
%     sub = [4*p+4*m+3*ntr+nval+7];
%     for i = 1:2*ntr+2*m+2
%         sub = [sub,4*p+2*m+3+ntr+nval+i];
%     end
%     sub = [sub,3*p+2*m+ntr+3,4*p+4*m+3*ntr+nval+6];
%     for j = 1:nval
%         sub = [sub,3*p+2*m+ntr+3+j];
%     end
%     for i = 1:m
%         sub = [sub,p+i];
%         for j = 1:gs
%             sub = [sub,(i-1)*gs+j];
%         end
%     end
%     for i = 1:m
%         sub = [sub,p+m+1+i];
%         for j = 1:gs
%             sub = [sub,p+2*m+2+(i-1)*gs+j];
%         end
%     end
% 
%     subptr = [1,2*ntr+2*m+4];% 2*ntr+2*m+nval+5
%     for k = 1:2*m
%         subptr = [subptr,2*ntr+2*m+nval+5+(k-1)*(gs+1)+1];
%     end

    prob.cones.sub = sub;
    prob.cones.subptr = subptr;

    [r,res] = mosekopt('minimize echo(0)',prob);
    
    moseksol = res.sol.itr.xx'; % row vector

    % update and record
    r0 = max(tol,moseksol(p+1:p+m+1))';
    lbd0 = max(tol,moseksol(p+m+2:p+2*m+2))';

    x_k = moseksol(1:p)';
    hist.x(iter,:) = x_k';
    hist.r(iter,:) = r0';
    hist.lbd(iter,:) = lbd0'; 

    hist.loss_val(iter) = 0.5/nval*norm(X_val*x_k-y_val)^2;
    hist.loss_test(iter) = 0.5/nte*norm(X_te*x_k-y_te)^2;
end
time = toc;
y = hist;
end


