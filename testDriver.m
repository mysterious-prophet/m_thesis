%% Test Driver
% runs testing configuration based on input configuration number for
% predefined configurations (see class configuration), outputs instance of
% result class containing results of experimental testing

% inputs: conf_num - number of predefined configuration
% outputs: results - instance of result class containing results of
%                    experimental testing

function results = testDriver(conf_num)  
    %% Load configuration, create instance of result class
    conf = configurationClass(conf_num);

    results = resultClass;

    pos_data_dir = getDataDirNameString(conf.pos_img_datastore.Folders);
    neg_data_dir = getDataDirNameString(conf.neg_img_datastore.Folders);
    fprintf("\n%s: Running testDriver: \n", getCurrentTime());
    fprintf("\t\t  Number of image(s) with positive trait: %d from folder: %s \n", conf.num_pos_imgs, pos_data_dir);
    fprintf("\t\t  Number of image(s) without positive trait: %d from folder: %s \n", conf.num_neg_imgs, neg_data_dir);


    %% Image feature calculation
    if(conf.calc_image_features)
        fprintf("%s: Calculating image feature(s)!\n", getCurrentTime());
        fprintf("\t\t  Binary mask threshold(s): %s\n", getArrayString(conf.bin_theta));
        fprintf("\t\t  Function(s): %s with max. n: %d and max. l: %d\n", getArrayString(conf.func_name), conf.n_max, conf.l_max);
        fprintf("\t\t  Rho(s): %s\n", getArrayString(conf.rho));
        fprintf("\t\t  Filter(s): %s\n", getArrayString(conf.filter_name));
        fprintf("\t\t  Kernel(s): %s\n", getArrayString(conf.kernel_style));
        fprintf("\t\t  Nonlinear transformation(s): %s\n", getArrayString(conf.nonlin_trans));
        fprintf("\t\t  Feature(s): %s\n", getArrayString(conf.features));
        fprintf("%s: Calculating feature(s) of image(s) with positive trait!\n", getCurrentTime());

        tic;
        results.pos_imgs_features = featureCalculationDriver(conf.pos_img_datastore, ...
            conf.num_pos_imgs, conf.bin_theta, conf.spect_norm, conf.func_name, ...
            conf.l_max, conf.n_max, conf.rho, conf.frame_style, conf.kernel_style, conf.filter_name, conf.nonlin_trans, conf.features);
        pos_time = toc;

        [pos_time_hs, pos_time_mins, pos_time_secs] = calcOperationTime(pos_time);
        fprintf("%s: Feature(s) of image(s) with positive trait calculated in %d hours %d minutes %d seconds!\n", getCurrentTime(), pos_time_hs, pos_time_mins, pos_time_secs);
        fprintf("%s: Calculating feature(s) of image(s) without positive trait!\n", getCurrentTime());

        tic;
        results.neg_imgs_features = featureCalculationDriver(conf.neg_img_datastore, ...
            conf.num_neg_imgs, conf.bin_theta, conf.spect_norm, conf.func_name, ...
            conf.l_max, conf.n_max, conf.rho, conf.frame_style, conf.kernel_style, conf.filter_name, conf.nonlin_trans, conf.features);
        neg_time = toc;

        [neg_time_hs, neg_time_mins, neg_time_secs] = calcOperationTime(neg_time);
        fprintf("%s: Feature(s) of image(s) without positive trait calculated in %d hours %d minutes %d seconds!\n", getCurrentTime(), neg_time_hs, neg_time_mins, neg_time_secs);
    end


    %% Feature statistical testing
    if(~conf.calc_image_features && conf.test_stat_importance)
        fprintf("%s: Loading data from folder Image Features!\n", getCurrentTime());
        results.pos_imgs_features = load(conf.X_train_filename);
        results.pos_imgs_features = results.pos_imgs_features.data;
        results.neg_imgs_features = load(conf.X_train_whit_filename);
        results.neg_imgs_features = results.neg_imgs_features.data;
    end

    if(conf.test_stat_importance)
        fprintf("%s: Testing statistical significance of calculated feature(s)!\n", getCurrentTime());
        fprintf("\t\t  Alpha, p-value: %d\n", conf.alpha);
        log_string = ["false" "true"];
        fprintf("\t\t  FDR refinement: %s\n", log_string(conf.fdr_refine+1));

        tic;
        [results.pos_imgs_features_signif, results.neg_imgs_features_signif, results.ranksum_stat_importance_all,...
            results.t_stat_importance_all, results.select_num_hyps_false_corr] = ...
            statTestingDriver(conf.func_name, conf.bin_theta, conf.rho, conf.filter_name, conf.kernel_style,  ...
            conf.nonlin_trans, conf.n_max, conf.l_max, conf.features, results.pos_imgs_features, results.neg_imgs_features, conf.alpha, conf.fdr_refine);
        test_time = toc;

        [test_time_hs, test_time_mins, test_time_secs] = calcOperationTime(test_time);
        fprintf("%s: Feature(s) tested in %d hours %d minutes %d seconds!\n", getCurrentTime(), test_time_hs, test_time_mins, test_time_secs);
    end
    
    %% PCA datawhitening of image features
    if(conf.whiten_data)
        fprintf("%s: Data whitening calculated feature(s)!\n", getCurrentTime());
        fprintf("\t\t  Number of features after data whitening: %s\n", getArrayString(conf.num_whit_features));

        tic;
        [results.X_train, results.X_train_whit] = dataWhiteningDriver(results.pos_imgs_features, results.neg_imgs_features, ...
            conf.func_name, conf.bin_theta, conf.rho, conf.filter_name, conf.kernel_style, conf.nonlin_trans, ...
            conf.n_max, conf.l_max, conf.features, conf.num_whit_features);
        whiten_time = toc;

        [whiten_time_hs, whiten_time_mins, whiten_time_secs] = calcOperationTime(whiten_time);
        fprintf("%s: Features whitened in %d hours %d minutes %d seconds!\n", getCurrentTime(), whiten_time_hs, whiten_time_mins, whiten_time_secs); 
    end

    %% Image classification
    if(~conf.whiten_data && conf.class_images)
        fprintf("%s: Loading data from folder Data Whitening!\n", getCurrentTime());
        results.X_train = load(conf.X_train_filename);
        results.X_train = results.X_train.data;
        results.X_train_whit = load(conf.X_train_whit_filename);
        results.X_train_whit = results.X_train_whit.data;
    end

    if(conf.class_images)
        fprintf("%s: Running classifier(s) on feature(s) without data whitening!\n", getCurrentTime());
        fprintf("\t\t  Classifier(s): %s\n", erase(getArrayString(conf.classifier_train_name), "train"));
        tic;
        [results.class_stats_leave1, results.class_stats_stratkfold, ...
            results.class_acc_leave1_lda,  results.class_acc_leave1_qda,  results.class_acc_leave1_knn, results.knn_opt, ~, ~, ~] = ...
            classificationDriver(results.X_train, size(results.X_train, 2), conf.classifier_train_name, conf.classifier_class_name, ...
            conf.k_fold, conf.train_params, conf.num_pos_imgs, conf.num_neg_imgs);
        class_time_sans = toc;
        [class_time_sans_hs, class_time_sans_mins, class_time_sans_secs] = calcOperationTime(class_time_sans);
        fprintf("%s: Image(s) without data whitening classified in %d hours %d minutes %d seconds!\n", getCurrentTime(), class_time_sans_hs, class_time_sans_mins, class_time_sans_secs);


        conf.train_params_triv = conf.setLearnParamsTriv(results.knn_opt);
        fprintf("%s: Running classifier(s) on single feature vectors!\n", getCurrentTime());
        fprintf("\t\t  Classifier(s): %s, with opt. num. of neighbours: %d\n", erase(getArrayString(conf.classifier_train_name_triv), "train"), results.knn_opt);
        tic;
        [results.class_stats_leave1_triv, results.signif_feats_class_stats_leave1, results.class_stats_stratkfold_triv] = ...
                classificationDriverTriv(results.X_train, conf.classifier_train_name_triv, conf.classifier_class_name_triv, ...
                conf.k_fold, conf.train_params_triv, conf.num_pos_imgs, conf.num_neg_imgs, conf.func_name, conf.bin_theta, conf.rho, ...
                conf.filter_name, conf.kernel_style, conf.nonlin_trans, conf.n_max, conf.l_max, conf.features);
        class_time_triv = toc;
        [class_time_triv_hs, class_time_triv_mins, class_time_triv_secs] = calcOperationTime(class_time_triv);
        fprintf("%s: Image(s) with single-feature vector data classified in %d hours %d minutes %d seconds!\n", getCurrentTime(), class_time_triv_hs, class_time_triv_mins, class_time_triv_secs);


        fprintf("%s: Running classifier(s) on feature(s) with data whitening!\n", getCurrentTime());
        fprintf("\t\t  Classifier(s): %s\n", erase(getArrayString(conf.classifier_train_name_whit), "train"));
        tic;
        [results.class_stats_leave1_whit, results.class_stats_stratkfold_whit, ...
            results.class_acc_leave1_lda_whit,  results.class_acc_leave1_qda_whit,  results.class_acc_leave1_knn_whit, ~,...
            results.class_acc_leave1_svm_whit, results.class_acc_leave1_sigmoid_whit, results.class_acc_leave1_tanh_whit] = ...
            classificationDriver(results.X_train_whit, conf.num_whit_features, conf.classifier_train_name_whit, conf.classifier_class_name_whit, ...
            conf.k_fold, conf.train_params_whit, conf.num_pos_imgs, conf.num_neg_imgs);
        class_time_whit = toc;
        [class_time_whit_hs, class_time_whit_mins, class_time_whit_secs] = calcOperationTime(class_time_whit);
        fprintf("%s: Image(s) with data whitening classified in %d hours %d minutes %d seconds!\n", getCurrentTime(), class_time_whit_hs, class_time_whit_mins, class_time_whit_secs);
    end

    %% Save results
    fprintf("%s: Saving results!\n", getCurrentTime());
    tic;
    saveResults(conf, results);
    save_time = toc;
    [save_time_hs, save_time_mins, save_time_secs] = calcOperationTime(save_time);
    fprintf("%s: Results saved in %d hours %d minutes %d seconds!\n", getCurrentTime(), save_time_hs, save_time_mins, save_time_secs);
end

%% Get Current Time
% get current time string for console info
function time_string = getCurrentTime()
    t = datetime("now");
    time_string = strcat(getTimeUnitString(hour(t)), ":", getTimeUnitString(minute(t)), ":", getTimeUnitString(second(t)));
end

% Get Time Unit String
% if e.g. seconds lower than 10 Matlab returns number without zero
% this function corrects this
function time_unit_string = getTimeUnitString(time_unit)
    if(round(time_unit) < 10)
        time_unit_string = strcat("0", string(round(time_unit)));
    else
        time_unit_string = string(round(time_unit));
    end
end

%% Get Data Directory Name
% get name of the directory containing image date for console info output
function data_dir_name = getDataDirNameString(data_dir)
    data_dir = string(data_dir);
    slash_ind = strfind(data_dir, "\");
    slash_ind = slash_ind(end-1);
    data_dir_name = extractBetween(data_dir, slash_ind+1, strlength(data_dir));
end

%% Get Array String
% see function getArrayString
% this function specializes and supersedes this function for case of
% testDriver because of ", " on line 3
function array_string = getArrayString(array_name)
    array_string = "";
    for i = 1:size(array_name, 2)
        array_string = strcat(array_string, string(array_name(i)), ", ");
    end
    array_string = extractBetween(array_string, 1, strlength(array_string) - 2);
end

%% Calculate Operation Time
% calculate elapsed time in hours, mins, secs
function [time_hs, time_mins, time_secs] = calcOperationTime(time)
    time = round(time);
    time_hs = floor(time / 3600);
    time_mins = floor((time - 3600*time_hs) / 60);
    time_secs = time - (3600*time_hs + 60*time_mins);
end
    