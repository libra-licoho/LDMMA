function data = generate_sgl_data(settings)
p = settings.num_features;
m = settings.num_group;
ntr = settings.num_train;
nval = settings.num_validate;
nte = settings.num_test;
n = ntr+nval+nte;
group_size = p/m;
base_size = p/3;
mu = zeros(p,1);
X = mvnrnd(mu,eye(p,p),n);

% thr = floor(5.0);
temp = 1:5;
temp = temp';
n_nonzero = 5;
base = [temp;zeros(base_size-n_nonzero,1)];
x_true = repmat(base,[3,1]);
y_true = X*x_true;

snr = 2;
epsilon = mvnrnd(zeros(n,1),eye(n,n),1);
SNR_factor = snr / norm(y_true) * norm(epsilon);
y = y_true+1.0/SNR_factor * epsilon';

data.X = X;
data.y = y;

end

