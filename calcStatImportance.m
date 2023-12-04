%% Calculate Statistical Importance
% calculates statistical importance of functions, thetas, rhos, filters,
% kernels, nonlinear transformations, invariants, features and whole
% configurations

% inputs: test_h - binary hypothesis matrix of chosen statistical test
%       : func_name - function for invariant calculation
%       : bin_theta - binary mask threshold
%       : rho - function radius
%       : filter_name - invariant filter
%       : kernel_style - invariant filter kernel
%       : nonlin_trans - nonlinear transformation applied on filtered
%                       invariant
%       : n_max - invariant max. n
%       : l_max - invariant max. l
%       : features - features calculated on invariant
% outputs: stat_importance_* - statistical importance of functions, thetas,
%                              rhos, filters, kernels, transformations, 
%                              invariants, features and whole
%                              configurations

function[stat_importance_func, stat_importance_theta, stat_importance_rho, ...
    stat_importance_filt, stat_importance_ker, stat_importance_tran, ...
    stat_importance_invar, stat_importance_feature, stat_importance_all] = ...
        calcStatImportance(test_h, func_name, bin_theta, rho, filter_name, kernel_style, nonlin_trans, n_max, l_max, features)
    
    [num_funcs, num_thetas, num_rhos, num_filts, num_kers, num_trans, num_invars, num_features, num_hyps] = ...
        getNums(func_name, bin_theta, rho, filter_name, kernel_style, nonlin_trans, n_max, l_max, features);

    stat_importance_invar = zeros(num_invars, 2);
    stat_importance_feature = zeros(num_features, 2);
    stat_importance_tran = zeros(num_trans, 2);
    stat_importance_ker = zeros(num_kers, 2);
    stat_importance_filt = zeros(num_filts, 2);
    stat_importance_rho = zeros(num_rhos, 2);
    stat_importance_theta = zeros(num_thetas, 2);
    stat_importance_func = zeros(num_funcs, 2);
    stat_importance_all = zeros(num_funcs, num_thetas, num_rhos, num_filts, num_kers, num_trans, 2);
    stat_names_all = strings(num_funcs, num_thetas, num_rhos, num_filts, num_kers, num_trans, 1);

    for i = 1:num_funcs
        for j = 1:num_thetas
            for k = 1:num_rhos
                for l = 1:num_filts
                    if(strcmp(filter_name(l), 'noFilt'))
                        for p = 1:num_invars
                            for r = 1:num_features
                                stat_importance_feature(r, 1) = stat_importance_feature(r, 1) + test_h(i, j, k, l, 1, 1, p, r);
                            end
                            stat_importance_invar(p, 1) = stat_importance_invar(p, 1) + sum(test_h(i, j, k, l, 1, 1, p, :));
                        end
                        stat_importance_tran(1, 1) = stat_importance_tran(1, 1) + sum(test_h(i, j, k, l, 1, 1, :, :), 'all');
                        stat_importance_ker(1, 1) = stat_importance_ker(1, 1) + sum(test_h(i, j, k, l, 1, :, :,  :), 'all');
                        stat_importance_all(i, j, k, l, 1, 1, 1) = stat_importance_all(i, j, k, l, 1, 1, 1) + sum(test_h(i, j, k, l, 1, 1, :,  :), 'all');
                        stat_names_all(i, j, k, l, 1, 1, 1) = append(func_name(i), ', ', string(bin_theta(j)), ', ', string(rho(k)), ', ', filter_name(l), ', ', kernel_style(1), ', ', nonlin_trans(1));
                    else
                        for m = 2:num_kers
                            for n = 1:num_trans
                                for p = 1:num_invars
                                    for r = 1:num_features
                                        stat_importance_feature(r, 1) = stat_importance_feature(r, 1) + test_h(i, j, k, l, m, n, p, r);
                                    end
                                    stat_importance_invar(p, 1) = stat_importance_invar(p, 1) + sum(test_h(i, j, k, l, m, n, p, :));
                                end
                                stat_importance_tran(n, 1) = stat_importance_tran(n, 1) + sum(test_h(i, j, k, l, m, n, :, :), 'all');
                                stat_importance_all(i, j, k, l, m, n, 1) = stat_importance_all(i, j, k, l, m, n, 1) + sum(test_h(i, j, k, l, m, n, :, :), 'all');
                                stat_names_all(i, j, k, l, m, n, 1) = append(func_name(i), ', ', string(bin_theta(j)), ', ', string(rho(k)), ', ', filter_name(l), ', ', kernel_style(m), ', ', nonlin_trans(n));
                            end
                            stat_importance_ker(m, 1) = stat_importance_ker(m, 1) + sum(test_h(i, j, k, l, m, :, :,  :), 'all');
                        end
                    end
                    stat_importance_filt(l, 1) = stat_importance_filt(l, 1) + sum(test_h(i, j, k, l, :, :, :, :), 'all', 'omitnan');
                end
                stat_importance_rho(k, 1) = stat_importance_rho(k, 1) + sum(test_h(i, j, k, :, :, :, :, :), 'all', 'omitnan');
            end
            stat_importance_theta(j, 1) = stat_importance_theta(j, 1)  + sum(test_h(i, j, :, :, :, :, :, :), 'all', 'omitnan');
        end
        stat_importance_func(i, 1) = sum(test_h(i, :, :, :, :, :, :, :), 'all', 'omitnan');
    end

    variable_names = ["Absolute incid.", "Relative incid."];
    stat_importance_func = arr2SortTab(stat_importance_func, num_hyps, num_funcs, variable_names, string(func_name));
    stat_importance_theta = arr2SortTab(stat_importance_theta, num_hyps, num_thetas, variable_names, string(bin_theta));
    stat_importance_rho = arr2SortTab(stat_importance_rho, num_hyps, num_rhos, variable_names, string(rho));

    stat_importance_nofilt = arr2SortTab(stat_importance_filt(1, :), num_hyps, num_hyps / (num_funcs * num_thetas * num_rhos * num_invars * num_features), variable_names, string(filter_name(1, 1)));
    if(num_filts == 1)
        stat_importance_filt = stat_importance_nofilt;
    else
        stat_importance_pfilt = arr2SortTab(stat_importance_filt(2:end, :), num_hyps, num_hyps / ((num_kers-1)*num_trans*num_rhos*num_thetas*num_funcs*num_invars*num_features), variable_names, string(filter_name(1, 2:end)));
        stat_importance_filt = [stat_importance_nofilt; stat_importance_pfilt];
    end
    stat_importance_filt = sortrows(stat_importance_filt, 'Relative incid.', 'descend');

    stat_importance_noker = arr2SortTab(stat_importance_ker(1, :), num_hyps, num_hyps / (num_funcs*num_thetas*num_rhos*num_invars*num_features), variable_names, string(kernel_style(1, 1)));
    if(num_kers == 1)
        stat_importance_ker = stat_importance_noker;
    else
        stat_importance_ftker = arr2SortTab(stat_importance_ker(2:end, :), num_hyps, num_hyps / (num_trans*(num_filts-1)*num_rhos*num_thetas*num_funcs*num_invars*num_features), variable_names, string(kernel_style(1, 2:end)));
        stat_importance_ker = [stat_importance_noker; stat_importance_ftker];
    end
    stat_importance_ker = sortrows(stat_importance_ker, 'Relative incid.', 'descend');

    stat_importance_notran = arr2SortTab(stat_importance_tran(1, :), num_hyps, num_hyps / (((num_filts-1)*(num_kers-1) + 1)*num_rhos*num_thetas*num_funcs*num_invars*num_features), variable_names, string(nonlin_trans(1, 1)));
    if(num_trans == 1)
        stat_importance_tran = stat_importance_notran;
    else
        stat_importance_logtran = arr2SortTab(stat_importance_tran(2, :), num_hyps, num_hyps / ((num_filts -1)*(num_kers - 1)*num_rhos*num_thetas*num_funcs*num_invars*num_features), variable_names, string(nonlin_trans(1, 2)));
        stat_importance_tran = [stat_importance_notran; stat_importance_logtran];
    end
    stat_importance_tran = sortrows(stat_importance_tran, 'Relative incid.', 'descend');

    stat_importance_feature = arr2SortTab(stat_importance_feature, num_hyps, num_features, variable_names, string(features));

    invar_names = strings(num_invars, 1);
    for k = 1:num_invars
        n_name = string(floor(k/(n_max+1) - 0.01));
        l_name = string(rem(k-1, l_max+1));
        invar_names(k) = append(n_name, ', ', l_name);
    end
    invar_names = string(invar_names);
    stat_importance_invar = arr2SortTab(stat_importance_invar, num_hyps, num_invars, variable_names, invar_names);

    stat_importance_all_1 = reshape(stat_importance_all(:, :, :, :, :, :, 1), 1, []);
    stat_importance_all_2 = reshape(stat_importance_all(:, :, :, :, :, :, 2), 1, []);
    stat_names_all = reshape(stat_names_all(:, :, :, :, :, :, 1), 1, []);
    stat_importance_all_1(stat_names_all == '') = [];
    stat_importance_all_2(stat_names_all == '') = [];
    stat_names_all(stat_names_all == '') = [];
    stat_importance_all = [stat_importance_all_1; stat_importance_all_2]';
    stat_importance_all = arr2SortTab(stat_importance_all, num_hyps, num_hyps / (num_invars * num_features), variable_names, string(stat_names_all));
end

%% Array to Sorted Table
% create sorted table from statistical importance array

% inputs: stat_importance - array of statistically important hypotheses
%       : num_hyps - total number of hypotheses
%       : num_vars - num hypothesis corresponding to chosen calculation mode
%       : row_names - function, thetas, ... etc. names
% outputs: stat_importance - sorted table giving information on
%                            statistically important functions, thetas, ... etc. 

function stat_importance = arr2SortTab(stat_importance, num_hyps, num_vars, variable_names, row_names)
    stat_importance(:, 2) = stat_importance(:, 1) / (num_hyps / num_vars);
    stat_importance = array2table(stat_importance,'VariableNames', variable_names, 'RowNames', row_names);
    stat_importance = sortrows(stat_importance, 'Relative incid.', 'descend');
end

function[num_funcs, num_thetas, num_rhos, num_filts, num_kers, num_trans, num_invars, num_features, num_hyps] = ...
            getNums(func_name, bin_theta, rho, filter_name, kernel_style, nonlin_trans, n_max, l_max, features)
    num_funcs = size(func_name, 2);
    num_thetas = size(bin_theta, 2);
    num_rhos = size(rho, 2);
    num_filts = size(filter_name, 2);
    num_kers = size(kernel_style, 2);
    num_trans = size(nonlin_trans, 2);
    num_invars = (n_max + 1) * (l_max + 1);
    num_features = size(features, 2); 
    num_hyps = num_funcs * num_thetas * num_rhos * ...
        ((num_filts - 1) * (num_kers - 1) * num_trans + 1) * num_invars * num_features;
end