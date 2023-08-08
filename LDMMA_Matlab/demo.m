clear all;
experiment = 2;
method = 1;
num_repeat = 1;
for i = 1:num_repeat
    if experiment == 1 
        rng(42);
        settings = settingnum(experiment);
        opts = optset(experiment,settings,method);
        data = generate_elastic_data(settings);
        if method == 1
            [y_ela,time_ela] = yalmip_elastic(settings,data,opts);
        elseif method == 2
            [y_ela,time_ela] = MOSEK_elastic(settings,data,opts);
        end
    elseif experiment == 2 
        settings = settingnum(experiment);
        opts = optset(experiment,settings,method);
        data = generate_sgl_data(settings);
        if method ==1
            [y_sgl,time_sgl] = yalmip_sgl(settings,data,opts);
        elseif method == 2
            [y_sgl,time_sgl] = MOSEK_sgl(settings,data,opts);
        end
    elseif experiment==3
        rng(42);
        CV = 3;
        dataset = 6;
        [data,settings] = generate_svm_data(CV,dataset);
        opts = optset(experiment,settings,method);
        if method == 1
            [y_svm,time_svm] = yalmip_svm(settings,data,opts);
        elseif method == 2
            [y_svm,time_svm] = MOSEK_svm(settings,data,opts);
        end
    end
end
