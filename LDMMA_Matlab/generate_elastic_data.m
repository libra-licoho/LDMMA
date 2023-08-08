function data = generate_elastic_data(settings)
data = struct();
p = settings.num_features;
ntr = settings.num_train;
nval = settings.num_validate;
nte = settings.num_test;
n = ntr+nval+nte;

mu = zeros(p,1);
temp = 1:p;
Sigma = -abs(ones(p,1)*temp-temp'*ones(1,p));
Sigma = 2.^Sigma;
X = mvnrnd(mu,Sigma,n);

n_nonzero = 15;
x_true = [ones(n_nonzero,1);zeros(p-n_nonzero,1)];
ind = randperm(size(x_true,1));
x_true = x_true(ind);
% x_true = randperm(x_true);
y_true= X*x_true;

snr = 2;
epsilon = mvnrnd(zeros(n,1),eye(n,n),1);
SNR_factor = snr / norm(y_true) * norm(epsilon);
y = y_true+1.0/SNR_factor * epsilon';
data.X = X;
data.y = y;

end
