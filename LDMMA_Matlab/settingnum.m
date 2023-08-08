function settings = settingnum(experiment)
settings = struct();
if experiment == 1
    settings.num_features = 250;
    settings.num_train = 100;
    settings.num_validate = 20;
    settings.num_test = 250;
elseif experiment == 2
    settings.num_features = 600;
    settings.num_train = 100;
    settings.num_validate = 100;
    settings.num_test = 100;
    settings.num_group = 30;
end
end