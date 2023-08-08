function opts = optset(experiment,settings,method)
opts = struct();
if experiment == 1
    if method == 1
        opts.lbd0 = 0.01*ones(2,1);
        opts.r0 = 12*ones(2,1);
        opts.epsilon = 5;
        opts.miter = 10;
    elseif method == 2
        opts.lbd0 = 0.01*ones(2,1);
        opts.r0 = 12*ones(2,1);
        opts.epsilon = 1e-3;
        opts.miter = 10;
        opts.tol = 1e-3;
    end
elseif experiment == 2
    if method == 1
        opts.lbd0 = 0.1*ones(settings.num_group+1,1); 
        opts.r0 = 5*ones(settings.num_group+1,1);
        opts.epsilon = 1; 
        opts.miter = 10; 
    elseif method == 2
        opts.lbd0 = 0.01*ones(settings.num_group+1,1); 
        opts.r0 = 1*ones(settings.num_group+1,1);
        opts.epsilon = 1e-4;
        opts.miter = 20;
        opts.tol = 1e-5;
    end
elseif experiment == 3
    if method == 1
        opts.lbd0 = 0.1; 
        opts.r0 = 0.5; 
        opts.epsilon = 1; 
        opts.miter = 10; 
    elseif method ==2
        opts.lbd0 = 0.1; 
        opts.r0 = 0.5; 
        opts.epsilon = 1; 
        opts.miter = 10; 
    end
end
end