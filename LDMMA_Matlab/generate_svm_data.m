function [data,settings] = generate_svm_data(CV,dataset)
    [X,y] = readdataset(dataset);
    settings.num_CV = CV;
    yc = unique(y);
    y(y==yc(1))=-1;
    y(y==yc(2))=1;
    [n,p] = size(X);
    
    settings.num_features = p;
    num_one_fold = floor(n/(2*CV));
    nte = n-num_one_fold*CV;
    ntr = num_one_fold*(CV-1)*CV;
    nval = num_one_fold*CV;
    num_CV_samples = nval;
    settings.num_train = ntr;
    settings.num_validate = nval;
    settings.num_test = nte;
    settings.num_one_fold = num_one_fold;
    
    temp = 1:n;
    ind = randperm(size(temp,2));
    temp = temp(ind);
    index_CV = cell(CV,1);
    for i = 1:CV
        index_CV{i} = (i-1)*num_one_fold+1:i*num_one_fold;
    end
    
    tmp_list = 1:CV;
    tmp_list = [tmp_list(2:CV),tmp_list];
    
    index_train = cell(3,1);
    index_validate = cell(3,1);
    for i = 1:CV
        train_tmp = [];
        for j = tmp_list(i:i+CV-2)
            train_tmp = [train_tmp,index_CV{j}];
        end
        index_train{i} = train_tmp;
        index_validate{i} = index_CV{i};
    end
    
    X = X(temp,:);
    y = y(temp);
    
    data.X = X(1:num_CV_samples,:);
    data.X_test = X(num_CV_samples+1:num_CV_samples+nte,:);
    data.y = y(1:num_CV_samples);
    data.y_test = y(num_CV_samples+1:num_CV_samples+nte);
    
    data.iTr = index_train;
    data.iVal = index_validate;

end
function [X,y] = readdataset(dataset)
    if dataset == 1
        load("X_liver.mat");
        load("y_liver.mat");
        X = X_liver;
        y = Y_liver;
    elseif dataset == 2
        load("X_breast.mat");
        load("y_breast.mat");
        X = X_bre;
        y = Y_bre;
    elseif dataset == 3
        load("X_dia.mat");
        load("y_dia.mat");
        X = X_dia;
        y = Y_dia;
    elseif dataset == 4
        load("X_sonar.mat");
        load("y_sonar.mat");
        X = X_sonar;
        y = Y_sonar;
    elseif dataset == 5
        load("X_w1a.mat");
        load("y_w1a.mat");
        X = X_w1a;
        y = Y_w1a;
    elseif dataset == 6
        load("X_a1a.mat");
        load("y_a1a.mat");
        X = X_a1a;
        y = Y_a1a;
    end
end