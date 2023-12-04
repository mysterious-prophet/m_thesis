%% One-Dimensional Classification Driver
% driver for testing one-feature classifiers

% inputs: X_train - training dataset
%       : num_whit_features - number of features in input data
%       : classifier_train_name - train using...
%       : classifier_class_name - classify using...
%       : k_fold - number of samples to be taken out of dataset for
%                  stratified-k-fold cross-validation
%       : train_params - classifier training parameters
%       : num_ad_imgs - number of images with positive trait
%       : num_cn_imgs - number of images without positive trait
%       : func_name - function used for invariant calculation
%       : bin_theta - binary mask threshold
%       : rho - function radius
%       : filter_name - names of used filters
%       : kernel_style - names of used filtering kernels
%       : nonlin_trans - names of used nonlinear transformations
%       : n_max - invariant max. n 
%       : l_max - invariant max. l
%       : features - names of calculated features
% outputs: class_stats_leave1 - classification statistics using
%                               leave-one-out cross-validation
%        : signif_features_class_stats_leave1 - classification statistics
%                                               of invariant features
%                                               declared as significant in
%                                               statTestingDriver
%        : class_stats_stratkfold - classification statistics using
%                                   stratified-k-fold cross-validation


function [class_stats_leave1, signif_features_class_stats_leave1, class_stats_stratkfold] = ...
    classificationDriverTriv(X_train, classifier_train_name, classifier_class_name, ...
     k_fold, train_params, num_ad_imgs, num_cn_imgs, func_name, bin_theta, rho, ...
     filter_name, kernel_style, nonlin_trans, n_max, l_max, features)

    num_classifiers = size(classifier_class_name, 2);
    num_features = size(X_train, 2);
    class_stats_leave1 = zeros(19, num_classifiers * num_features);
    class_stats_stratkfold = zeros(19, num_classifiers * num_features);

    y_star_train = ones(num_ad_imgs + num_cn_imgs, 1);
    y_star_train(num_ad_imgs + 1:end) = y_star_train(num_ad_imgs + 1:end) - 1;
    for i = 1:num_classifiers 
        for j = 1:num_features
            [class_stats_leave1(:, (i-1)*num_features + j), ...
                class_stats_stratkfold(:, (i-1)*num_features + j)] = ...
                classImages(X_train(:, j), y_star_train, num_ad_imgs, num_cn_imgs, ...
                classifier_train_name(i), classifier_class_name(i), k_fold, train_params{1, i});
        end
    end

    [~, sort_indcs_se_star_acc_leave1] = sortrows(class_stats_leave1', [3 14], ["descend" "descend"]);
    sort_indcs_se_star_acc_leave1 = sort_indcs_se_star_acc_leave1';
    class_stats_leave1 = class_stats_leave1(:, sort_indcs_se_star_acc_leave1);
    [~, sort_indcs_se_star_acc_stratkfold] = sortrows(class_stats_stratkfold', [3 14], ["descend" "descend"]);
    sort_indcs_se_star_acc_stratkfold = sort_indcs_se_star_acc_stratkfold';

    class_stats_leave1(3, :) = [];
    class_stats_stratkfold(3, :) = [];

    var_names = createVarNames(func_name, bin_theta, rho, filter_name, kernel_style, nonlin_trans, n_max, l_max, features);
    var_names = reshape(var_names, 1, []);
    var_names = reduceVarNames(var_names, filter_name, kernel_style);
    var_names = var_names(sort_indcs_se_star_acc_leave1);

    signif_filename = strcat('AD_Images_Signif._Feats._Num._AD=', string(num_ad_imgs), '_Num._CN=', string(num_cn_imgs), '_Theta=', getArrayString(bin_theta), ...
            '_Func(s).=', getArrayString(func_name), '_nMax=', string(n_max), '_lMax=', string(l_max), '_Rho=', getArrayString(rho), ...
            '_Filter(s)=', getArrayString(filter_name) ,'_Kernel(s)=', getArrayString(kernel_style), '.csv');
    signif_table = readtable(strcat('results/Statistical Significance/', signif_filename));
    signif_feats_names = signif_table.Properties.VariableDescriptions;
    signif_features_class_stats_leave1 = zeros(18, size(signif_feats_names, 2));
    signif_features_var_names = strings(1, size(signif_feats_names, 2));
    j = 1;
    for i = 1:num_features
        if(ismember(var_names(i), signif_feats_names))
            signif_features_class_stats_leave1(:, j) = class_stats_leave1(:, i);
            signif_features_var_names(j) = var_names(i);
            j = j + 1;
        end
    end
    signif_features_class_stats_leave1 = createClassStatsTable(signif_features_class_stats_leave1, signif_features_var_names);

    class_stats_leave1 = createClassStatsTable(class_stats_leave1, var_names);
    if(num_ad_imgs == num_cn_imgs && rem(k_fold, 2) == 0)
        class_stats_stratkfold = createClassStatsTable(class_stats_stratkfold(:, sort_indcs_se_star_acc_stratkfold), var_names(sort_indcs_se_star_acc_stratkfold));
    end
end

%% Reduce Variable Names
% we have too many variable names combinations, e.g. we cannot have
% filter_name = noFilt and kernel_style = gaussKer, we must thus reduce the
% number of variables

% inputs: var_names - variable names array
%       : filter_name - names of used filters
%       : kernel_style - styles of used kernels
% outputs: var_names - variable names array rid of impossible combinations

function var_names = reduceVarNames(var_names, filter_name, kernel_style)
    num_vars = size(var_names, 2);
    num_filts = size(filter_name, 2);
    num_kers = size(kernel_style, 2);
    for i = 1:num_vars 
        for j = 1:num_filts
            for k = 1:num_kers
                if(contains(var_names(i), 'noFilt') && ~strcmp(kernel_style(k), 'noKer') && contains(var_names(i), kernel_style(k))...
                        || ~contains(var_names(i), 'noFilt') && strcmp(kernel_style(k), 'noKer') && contains(var_names(i), kernel_style(k)))
                    var_names(i) = '';
                end
            end
        end
    end
    var_names(var_names == '') = [];
end