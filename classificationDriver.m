%% Classification Driver
% driver for image classification based on invariant features

% inputs: X_train - training dataset
%       : num_whit_features - number of features in input data
%       : classifier_train_name - train using...
%       : classifier_class_name - classify using...
%       : k_fold - number of samples to be taken out of dataset for
%                  stratified-k-fold cross-validation
%       : train_params - classifier training parameters
%       : num_pos_imgs - number of images with positive trait
%       : num_neg_imgs - number of images without positive trait
% outputs: class_stats_leave1 - results of leave one out cross-validation
%        : class_stat_stratkfold - resullts of stratified k fold
%                                  cross-validation
%        : class_acc_<> - figure containing plot of best classifications for
%                        given classifier 

function [class_stats_leave1, class_stats_stratkfold, ...
     class_acc_leave1_lda,  class_acc_leave1_qda,  class_acc_leave1_knn, knn_opt, class_acc_leave1_svm, ...  
     class_acc_leave1_sigmoid, class_acc_leave1_tanh] = ...
    classificationDriver(X_train, num_whit_features, classifier_train_name, classifier_class_name, ...
        k_fold, train_params, num_pos_imgs, num_neg_imgs) 

    num_num_whit_features = size(num_whit_features, 2);
    num_classifiers = size(classifier_class_name, 2);
    max_num_train_configs = getNumLearnConfigs(train_params);

    y_star_train = ones(num_pos_imgs + num_neg_imgs, 1);
    y_star_train(num_pos_imgs + 1:end) = y_star_train(num_pos_imgs + 1:end) - 1;

    class_stats_leave1 = zeros(19, num_num_whit_features * num_classifiers * max_num_train_configs);
    class_stats_stratkfold = zeros(19, num_num_whit_features * num_classifiers * max_num_train_configs);

    var_names = strings(1, num_num_whit_features * num_classifiers * max_num_train_configs);
    for i = 1:num_num_whit_features
        for j = 1:num_classifiers
            for k = 1:size(train_params{1, j}, 2)
                if(num_num_whit_features > 1)
                    X_train_temp = squeeze(X_train(i, :, 1:num_whit_features(i)));
                else
                    X_train_temp = X_train;
                end
                [class_stats_leave1(:, (i-1)*(num_classifiers*max_num_train_configs) + (j-1)*max_num_train_configs + k), ...
                    class_stats_stratkfold(:, (i-1)*(num_classifiers*max_num_train_configs) + (j-1)*max_num_train_configs + k)] = ...
                    classImages(X_train_temp, y_star_train, num_pos_imgs, num_neg_imgs, ...
                    classifier_train_name(j), classifier_class_name(j), k_fold, train_params{1, j}{k});
                
                var_name = getVarName(classifier_train_name(j), train_params{1, j}{k}, num_whit_features(i), num_num_whit_features);
                var_names(1, (i-1)*(num_classifiers*max_num_train_configs) + (j-1)*max_num_train_configs + k) = var_name;
            end
        end
    end

    class_stats_stratkfold = class_stats_stratkfold(:, any(class_stats_leave1));
    var_names = var_names(:, any(class_stats_leave1));
    class_stats_leave1 = class_stats_leave1(:, any(class_stats_leave1));

    
    [~, sort_indcs_se_star_acc_leave1] = sortrows(class_stats_leave1', [3 14], ["descend" "descend"]);
    sort_indcs_se_star_acc_leave1 = sort_indcs_se_star_acc_leave1';
    [~, sort_indcs_se_star_acc_stratkfold] = sortrows(class_stats_stratkfold', [3 14], ["descend" "descend"]);
    sort_indcs_se_star_acc_stratkfold = sort_indcs_se_star_acc_stratkfold';

    if(num_num_whit_features == 1)
            [lda_best_class, lda_best_pca_dims, ~] = findBestClass(var_names(sort_indcs_se_star_acc_leave1), class_stats_leave1(:, sort_indcs_se_star_acc_leave1), "LDA", 1, "\lambda");
            [qda_best_class, qda_best_pca_dims, ~] = findBestClass(var_names(sort_indcs_se_star_acc_leave1), class_stats_leave1(:, sort_indcs_se_star_acc_leave1), "QDA", 1, "\lambda");
            [knn_best_class, knn_best_pca_dims, knn_opt] = findBestClass(var_names(sort_indcs_se_star_acc_leave1), class_stats_leave1(:, sort_indcs_se_star_acc_leave1), "KNN", 1, "Neighbors");
            class_acc_leave1_svm = [];
            class_acc_leave1_sigmoid = [];
            class_acc_leave1_tanh = [];
    else
        [lda_best_class, lda_best_pca_dims, ~] = findBestClass(var_names(sort_indcs_se_star_acc_leave1), class_stats_leave1(:, sort_indcs_se_star_acc_leave1), "LDA", 2, "\lambda");
        [qda_best_class, qda_best_pca_dims, ~] = findBestClass(var_names(sort_indcs_se_star_acc_leave1), class_stats_leave1(:, sort_indcs_se_star_acc_leave1), "QDA", 2, "\lambda");
        [knn_best_class, knn_best_pca_dims, knn_opt] = findBestClass(var_names(sort_indcs_se_star_acc_leave1), class_stats_leave1(:, sort_indcs_se_star_acc_leave1), "KNN", 2, "Neighbors");
        [svm_best_class, svm_best_pca_dims, ~] = findBestClass(var_names(sort_indcs_se_star_acc_leave1), class_stats_leave1(:, sort_indcs_se_star_acc_leave1), "Gauss", 2, "\sigma");
        [sigmoid_best_class, sigmoid_best_pca_dims, ~] = findBestClass(var_names(sort_indcs_se_star_acc_leave1), class_stats_leave1(:, sort_indcs_se_star_acc_leave1), "sigmoid", 2, "Neurons");
        [tanh_best_class, tanh_best_pca_dims, ~] = findBestClass(var_names(sort_indcs_se_star_acc_leave1), class_stats_leave1(:, sort_indcs_se_star_acc_leave1), "tanh", 2, "Neurons");

        class_acc_leave1_svm = plotClassAcc(svm_best_class, svm_best_pca_dims, "SVM", "\sigma");
        class_acc_leave1_sigmoid = plotClassAcc(sigmoid_best_class, sigmoid_best_pca_dims, "ANN", "Number of Neurons", "sigmoid"); 
        class_acc_leave1_tanh = plotClassAcc(tanh_best_class, tanh_best_pca_dims, "ANN", "Number of Neurons", "tanh"); 
    end
    class_acc_leave1_lda = plotClassAcc(lda_best_class, lda_best_pca_dims, "LDA", "\lambda");
    class_acc_leave1_qda = plotClassAcc(qda_best_class, qda_best_pca_dims, "QDA", "\lambda");
    class_acc_leave1_knn = plotClassAcc(knn_best_class, knn_best_pca_dims, "KNN", "Number of Neighbors");

    class_stats_leave1(3, :) = [];
    class_stats_stratkfold(3, :) = [];

    class_stats_leave1 = createClassStatsTable(class_stats_leave1(:, sort_indcs_se_star_acc_leave1), var_names(sort_indcs_se_star_acc_leave1));
    if(rem(k_fold, 2) == 0)
        class_stats_stratkfold = createClassStatsTable(class_stats_stratkfold(:, sort_indcs_se_star_acc_stratkfold), var_names(sort_indcs_se_star_acc_stratkfold));
    end
end

%% Get Number of Configurations
% get number of classification configurations

% inputs: train_params - classifier training parameters
% outputs: max_num_train_configs - max. number of training configurations

function max_num_train_configs = getNumLearnConfigs(train_params)
    max_num_train_configs = 0;
    for i = 1:size(train_params, 2)
        if(size(train_params{1, i}, 2) > max_num_train_configs)
            max_num_train_configs = size(train_params{1, i}, 2);
        end
    end
end

%% Get Variable Name
% get variable name for table containing sorted classifiers

% inputs: classifier_train_name - name of trained classifier
%       : train_params - training parameters
%       : num_whit_features - number of input features
%       : num_num_whit_features - total number of input features variants
% outputs: var_name - variable name containing classifier and its params

function var_name = getVarName(classifier_train_name, train_params, num_whit_features, num_num_whit_features)
    if(num_num_whit_features > 1)
        data_string = "PCA dims.=";
    else
        data_string = "Feats.=";
    end
    var_name = classifier_train_name;
    var_name = erase(var_name, "train");
    if(strcmp(var_name, "LDA") || strcmp(var_name, "QDA"))
        class_pars_names = "\lambda";
    elseif(strcmp(var_name, "KNN"))
        class_pars_names = ["Neighbors", "Norm"];
    elseif(strcmp(var_name, "SVM"))
        if(train_params(1) == 0)
            class_pars_names = ["Gauss.", "Out. coeff.", "\nu", "\sigma"];
        else
            class_pars_names = ["Pol. order", "Out. coeff.", "\nu"];
        end
    elseif(strcmp(var_name, "ANN"))
        class_pars_names = ["Neurons", "Act.funcs.", "\lambda"];
    end
    if(~iscell(train_params(1, 1)))
        par_names = string(train_params);
        if(strcmp(class_pars_names(1), "Gauss."))
            par_name = class_pars_names(1);
        else
            par_name = strcat(class_pars_names(1), "=", par_names(1, 1));
        end
        if(size(par_names, 2) > 1)
            for i = 2:size(par_names, 2)
                par_name = strcat(par_name, ",", class_pars_names(i), "=", par_names(1, i));
            end
        end
    else
        num_params = size(train_params{1, 1}, 2);
        par_name = strings(1);
        for i = 1:size(train_params, 2) - 1
            for j = 1:num_params
                par_name = strcat(par_name, class_pars_names(i), "=", string(train_params{1, i}(j)), ",");
            end
        end
        par_name = strcat(par_name, class_pars_names(size(train_params, 2)), "=", string(train_params{1, size(train_params, 2)}(1)));
    end
    var_name = strcat(data_string , string(num_whit_features), ",", var_name, ":", par_name);
end


%% Find Best Classifier
% for each classifier type find the best instance 

% inputs: var_names_best -  variable names
%       : class_stats_leave1 - array containing classification statistics
%       : class_name - name of classifier type
%       : type - classification without or with data whitening
%       : par_name - name of classifier"s key parameter
% outputs: class_acc_leave1_best_sort - array containing parameter values
%                                       and corresponding with best
%                                       classification accuracy values
%        : feat_dim_all - dimension, for which best results were achieved
%        : knn_opt - optimum number of neigbors for knn classifier to be
%                    used for trivial classifiers 


function [class_acc_leave1_best_sort, feat_dim_all, knn_opt] = findBestClass(var_names_best, class_stats_leave1, class_name, type, par_name)
    class_empty_cell_indcs = cellfun(@isempty, strfind(var_names_best, class_name));
    class_best_ind = find(class_empty_cell_indcs == 0);
    knn_opt = 0;

    j = 2;
    class_dims_candidates = zeros(1, size(var_names_best, 2));
    class_dims_candidates(1) = getFeatDim(var_names_best(class_best_ind(1)));
    for i = class_best_ind(2:end)
        cur_dim = getFeatDim(var_names_best(class_best_ind(j)));
        if(ismember(cur_dim, class_dims_candidates))
            class_best_ind(j) = 0;
        end
        if(class_stats_leave1(3, i) < class_stats_leave1(3, class_best_ind(1)))
             class_best_ind(j:end) = 0;
             break;
        end
        class_dims_candidates(j) = cur_dim;
        j = j+1;
    end
    class_best_ind = class_best_ind(class_best_ind ~= 0);

    class_acc_leave1_best_sort = zeros(size(class_best_ind, 2), size(var_names_best, 2));
    class_se_star_leave1_best_sort = zeros(size(class_best_ind, 2), size(var_names_best, 2));
    feat_dim_all = zeros(size(class_best_ind, 2), 1);
    for i = 1:size(class_best_ind, 2)
        feat_dim_all(i, 1) = getFeatDim(var_names_best(class_best_ind(1, i)));
        if(type == 1)
            feat_empty_cells_indcs = cellfun(@isempty, strfind(var_names_best, strcat("Feats.=", string(feat_dim_all(i, 1)))));
        elseif(type == 2)
            feat_empty_cells_indcs = cellfun(@isempty, strfind(var_names_best, strcat("PCA dims.=", string(feat_dim_all(i, 1)))));
        end
        class_best_indcs = logical(~class_empty_cell_indcs .* ~feat_empty_cells_indcs);
        class_stats_leave1_best = class_stats_leave1(:, class_best_indcs);
        var_names_best_temp = var_names_best(class_best_indcs);
        par_names = strings(1, size(var_names_best_temp, 2));
        par_names(1, :) = par_name;
        var_names_best_temp = arrayfun(@extractAfter, var_names_best_temp, par_names);
        par_names(1, :) = "=";
        var_names_best_temp = arrayfun(@extractAfter, var_names_best_temp, par_names);
        if(contains(var_names_best_temp(1), "="))
            par_names(1, :) = ","; 
            var_names_best_temp = arrayfun(@extractBefore, var_names_best_temp, par_names);
        end
        var_names_best_temp = arrayfun(@str2num, var_names_best_temp);
        [var_names_best_sort, sort_indcs] = sort(var_names_best_temp, "ascend");
        class_acc_leave1_best = class_stats_leave1_best(14, :);
        class_se_star_leave1_best = class_stats_leave1_best(3, :);
        class_acc_leave1_best_sort(i, 1:size(var_names_best_temp, 2)) = class_acc_leave1_best(sort_indcs);
        class_se_star_leave1_best_sort(i, 1:size(var_names_best_temp, 2)) = class_se_star_leave1_best(sort_indcs);
    end
    class_acc_leave1_best_sort = class_acc_leave1_best_sort(:, 1:size(var_names_best_temp, 2));
    class_se_star_leave1_best_sort = class_se_star_leave1_best_sort(:, any(class_se_star_leave1_best_sort));
    mean_se_star = mean(class_se_star_leave1_best_sort, 2);
    max_se_star_indcs_temp = zeros(size(class_best_ind, 2), 1);
    max_se_star_indcs_temp(mean_se_star == max(mean_se_star)) = 1;
    class_acc_leave1_best_sort = class_acc_leave1_best_sort(logical(max_se_star_indcs_temp), :);
    feat_dim_all = feat_dim_all(logical(max_se_star_indcs_temp));
    if(strcmp(class_name, "KNN") && type == 1)
        knn_acc_opt_val = max(class_acc_leave1_best_sort, [], "all");
        [~, knn_opt_col] = find(class_acc_leave1_best_sort == knn_acc_opt_val);
        knn_opt = var_names_best_sort(1, knn_opt_col);
        knn_opt = knn_opt(1);
    end
    class_acc_leave1_best_sort = [var_names_best_sort; class_acc_leave1_best_sort];
end

% Get Number of Features
% finds number of features in variable string

% inputs: var_name_best - variable name of best instance of given
%                         classifier type
% ouputs: feat_dim - dimension, for which we obtained best classification
%                    results

function feat_dim = getFeatDim(var_name_best)
    feat_dim = extractBetween(var_name_best, "=", ",");
    feat_dim = str2double(feat_dim(1));
end

%% Plot Classification Accuracy
% plot classification accuracy of of best classifier instance

% inputs: class_acc_plot_arr - array containing parameter values and
%                              classification accuracy values for all best
%                              instances of given classifier type
%       : num_whit_features - number of input features
%       : class_name - name of the classifier type
%       : x_label - name of classifier key parameter
%       : varargin - additional parameter for ANN - hidden layer activation
%                    function name
% outputs: class_acc_plot - 2D plot of classification accuracy depending on
%                           key parameter value

function class_acc_plot = plotClassAcc(class_acc_plot_arr, num_whit_features, class_name, x_label, varargin)
    if(size(varargin, 2) > 0)
        add_params = varargin{1};
    end

    class_acc_plot = figure("Name", "Leave-One-Out Classification Accuracy");
    plot_x = class_acc_plot_arr(1, :);
    num_plots = size(class_acc_plot_arr, 1) - 1;
    plot_y = zeros(num_plots, size(class_acc_plot_arr, 2));
    plot_legend = strings(1, num_plots);
    for i = 1:num_plots
        plot_y(i, :) = class_acc_plot_arr(i+1, :);
        if(num_whit_features(i) > 10)
            if(strcmp(class_name, "LDA") || strcmp(class_name, "QDA") || ...
                    strcmp(class_name, "SVM"))
                semilogx(plot_x, plot_y(i, :), "-", "LineWidth", 1.25);
                hold on;
            else
                plot(plot_x, plot_y(i, :), "o", "LineWidth", 1.25);
                hold on;
            end
            plot_legend(1, i) = strcat("Features = ",  string(num_whit_features(i)));
        else
            equal_ind = isPlotEqual(class_acc_plot_arr, i+1);
            if(~isnan(equal_ind))
                plot_legend(1, equal_ind) = strcat(plot_legend(1, equal_ind), ", ", string(num_whit_features(i)));
            else
                if(strcmp(class_name, "LDA") || strcmp(class_name, "QDA") || ...
                        strcmp(class_name, "SVM"))
                    semilogx(plot_x, plot_y(i, :), "-", "LineWidth", 1.25);
                    hold on;
                else
                    plot(plot_x, plot_y(i, :), "o", "LineWidth", 1.25);
                    hold on;
                end
                plot_legend(1, i) = strcat("PCA Dims. = ", string(num_whit_features(i)));
            end
        end
    end

    if(strcmp(class_name, "ANN"))
        plot_legend = arrayfun(@strcat, plot_legend, ", Act. Func. = ", add_params(1));
    end

    hold off;
    ylim([0.5 1]);
    xlim([min(plot_x) max(plot_x)]);
    if(strlength(x_label) == 1)
        x_label = strcat("\it{", string(x_label), "}");
    end
    xlabel(x_label) ;
    ylabel("\it Accuracy");
    set(gca, "FontName", "Times");
    legend(plot_legend, "Location", "best", "FontName","Times");
end

% Check If Plot Is Equal
% check whether the current plot is totally equal to some previous plot, in
% such case we do not need to plot another line but merely add to previous
% plot legend

% inputs: class_acc_plot_arr - array containing parameter values and
%                              classification accuracy values for all best
%                              instances of given classifier type
%       : ind - index of current plot
% outputs: equal_ind - index of plots, which have the same values in all
%                      points

function equal_ind = isPlotEqual(class_acc_plot_arr, ind)
    equal_ind = NaN(size(class_acc_plot_arr, 1), 1) - 1;
    for i = 2:1:ind-1
        if(isequal(class_acc_plot_arr(i, :), class_acc_plot_arr(ind, :)))
            equal_ind(i, 1) = i;
        end
    end
    equal_ind = min(equal_ind) - 1;
end