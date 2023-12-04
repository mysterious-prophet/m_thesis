%% Statistical Testing Driver
% test statistical significance of calculated image features - using
% eigther Wilcoxon Ranksum test or T-test check, whether there are
% statistically significant differences between images with positive trait
% and images without positive trait

% inputs: func_name - function used for invariant calculation
%       : bin_theta - binary mask threshold
%       : rho - function radius
%       : filter_name - names of used filters
%       : kernel_style - names of used filtering kernels
%       : nonlin_trans - names of used nonlinear transformations
%       : n_max - invariant max. n 
%       : l_max - invariant max. l
%       : features - names of calculated features
%       : pos_imgs_features - calculated features of images with pos. trait
%       : neg_imgs_features - calculated features of images without pos.
%                             trait
%       : alpha - p-value threshold
%       : fdr_refinement - bool. determining whether to use
%                          Benjamini-Yekutieli False Discovery Rate
%                          refinement procedure
% outputs: pos_imgs_feats_signif - statistically significant features of
%                                  images with positive trait
%        : neg_imgs_feats_signif - statistically significant features of
%                                  images without positive trait
%        : stat_importance_all_rank - statistical importance of all
%                                     configurations rated acc. to ranksum
%        : stat_importance_all_t - statistical importance of all
%                                  configurations rated acc. to t-test
%        : select_num_hyps_false_corr - number of statistically significant
%                                       features after FDR refinement

function [pos_imgs_feats_signif, neg_imgs_feats_signif, stat_importance_all_rank,...
    stat_importance_all_t, select_num_hyps_false_corr] = ...
    statTestingDriver(func_name, bin_theta, rho, filter_name, kernel_style,  ...
        nonlin_trans, n_max, l_max, features, pos_imgs_feats, neg_imgs_feats, alpha, fdr_refine)

    [num_funcs, num_thetas, num_rhos, num_filts, num_kers, num_trans, num_invars, num_features, num_hyps] = ...
        getNums(func_name, bin_theta, rho, filter_name, kernel_style, nonlin_trans, n_max, l_max, features);

    pos_lillie_h = NaN(num_funcs, num_thetas, num_rhos, num_filts, num_kers, num_trans, num_invars, num_features);
    neg_lillie_h = NaN(num_funcs, num_thetas, num_rhos, num_filts, num_kers, num_trans, num_invars, num_features); 
    lilie_count_abs = 0;
    ranksum_p = NaN(num_funcs, num_thetas, num_rhos, num_filts, num_kers, num_trans, num_invars, num_features); 
    ranksum_h = NaN(num_funcs, num_thetas, num_rhos, num_filts, num_kers, num_trans, num_invars, num_features);
    t_p = NaN(num_funcs, num_thetas, num_rhos, num_filts, num_kers, num_trans, num_invars, num_features);
    t_h = NaN(num_funcs, num_thetas, num_rhos, num_filts, num_kers, num_trans, num_invars, num_features);

    for i = 1:num_funcs
        for j = 1:num_thetas
            for k = 1:num_rhos
                for l = 1:num_filts
                    if(strcmp(filter_name(l), 'noFilt'))
                        for p = 1:num_invars
                            for r = 1:num_features
                                [ranksum_p(i, j, k, 1, 1, 1, p, r), ranksum_h(i, j, k, 1, 1, 1, p, r)] = ranksum(pos_imgs_feats(:, i, j, k, l, 1, 1, p, r), neg_imgs_feats(:, i, j, k, l, 1, 1, p, r), 'alpha', alpha);
                                pos_lillie_h(i, j, k, 1, 1, 1, p, r) = lillietest(pos_imgs_feats(:, i, j, k, l, 1, 1, p, r));
                                neg_lillie_h(i, j, k, 1, 1, 1, p, r) = lillietest(neg_imgs_feats(:, i, j, k, l, 1, 1, p, r));
                                if(pos_lillie_h(i, j, k, 1, 1, 1, p, r) == 0 && neg_lillie_h(i, j, k, 1, 1, 1, p, r) == 0)
                                    [t_h(i, j, k, 1, 1, 1, p, r), t_p(i, j, k, 1, 1, 1, p, r)] = ttest2(pos_imgs_feats(:, i, j, k, l, 1, 1, p, r), neg_imgs_feats(:, i, j, k, l, 1, 1, p, r), 'Vartype', 'unequal', 'Alpha', alpha);
                                    lilie_count_abs = lilie_count_abs + 1;
                                else
                                    t_h(i, j, k, 1, 1, 1, p, r) = 0;
                                    t_p(i, j, k, 1, 1, 1, p, r) = 1;
                                end
                            end
                        end    
                    else
                        for m = 2:num_kers
                            for n = 1:num_trans
                                for p = 1:num_invars
                                    for r = 1:num_features
                                        [ranksum_p(i, j, k, l, m, n, p, r), ranksum_h(i, j, k, l, m, n, p, r)] = ranksum(pos_imgs_feats(:, i, j, k, l, m, n, p, r), neg_imgs_feats(:, i, j, k, l, m, n, p, r), 'alpha', alpha);
                                        pos_lillie_h(i, j, k, l, m, n, p, r) = lillietest(pos_imgs_feats(:, i, j, k, l, m, n, p, r));
                                        neg_lillie_h(i, j, k, l, m, n, p, r) = lillietest(neg_imgs_feats(:, i, j, k, l, m, n, p, r));
                                        if(pos_lillie_h(i, j, k, l, m, n, p, r) == 0 && neg_lillie_h(i, j, k, l, m, n, p, r) == 0)
                                            [t_h(i, j, k, l, m, n, p, r), t_p(i, j, k, l, m, n, p, r)] = ttest2(pos_imgs_feats(:, i, j, k, l, m, n, p, r), neg_imgs_feats(:, i, j, k, l, m, n, p, r), 'Vartype', 'unequal', 'Alpha', alpha);
                                        else
                                            t_h(i, j, k,  l, m, n, p, r) = 0;
                                            t_p(i, j, k, l, m, n, p, r) = 1;
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    lilie_count_rel = lilie_count_abs / num_hyps;

    [ranksum_stat_importance_func, ranksum_stat_importance_theta, ranksum_stat_importance_rho, ...
    ranksum_stat_importance_filt, ranksum_stat_importance_ker, ranksum_stat_importance_tran, ...
    ranksum_stat_importance_invar, ranksum_stat_importance_feature, ranksum_stat_importance_all] = ...
        calcStatImportance(ranksum_h, func_name, bin_theta, rho, filter_name, kernel_style, nonlin_trans, n_max, l_max, features);

    [t_stat_importance_func, t_stat_importance_theta, t_stat_importance_rho, ...
    t_stat_importance_filt, t_stat_importance_ker, t_stat_importance_tran, ...
    t_stat_importance_invar, t_stat_importance_feature, t_stat_importance_all] = ...
        calcStatImportance(t_h, func_name, bin_theta, rho, filter_name, kernel_style, nonlin_trans, n_max, l_max, features);

    stat_importance_all_rank = ranksum_stat_importance_all;
    stat_importance_all_t = t_stat_importance_all;

    [ranksum_p_sort, ~, ranksum_h_sort, ranksum_h_sort_nans, ~, ranksum_ind_nans] = sortTestResults(ranksum_p, ranksum_h);
    [t_p_sort, ~, ~, t_h_sort_nans, ~, t_ind_nans] = sortTestResults(t_p, t_h);
    [~, ranksum_final_ind] = sort(ranksum_ind_nans);
    [~, t_final_ind] = sort(t_ind_nans);

    ranksum_num_hyps_false = sum(ranksum_h_sort);
    if(fdr_refine && ranksum_num_hyps_false > 0)
        ranksum_num_hyps_false_corr = findPValThresholdInd(num_hyps, alpha, ranksum_p_sort) - 1;
    %     t_num_hyps_false = sum(t_h_sort);
        t_num_hyps_false_corr = findPValThresholdInd(num_hyps, alpha, t_p_sort) - 1;
    
        ranksum_h_sort_nans = refineFDR(ranksum_num_hyps_false_corr + 1, ranksum_h_sort_nans);
        ranksum_h_ref = reshape(ranksum_h_sort_nans(ranksum_final_ind), num_funcs, num_thetas, num_rhos, num_filts, num_kers, num_trans, num_invars, num_features);
        t_h_sort_nans = refineFDR(t_num_hyps_false_corr + 1, t_h_sort_nans);
        t_h_ref = reshape(t_h_sort_nans(t_final_ind), num_funcs, num_thetas, num_rhos, num_filts, num_kers, num_trans, num_invars, num_features);
    
            [ranksum_stat_importance_func_ref, ranksum_stat_importance_theta_ref, ranksum_stat_importance_rho_ref, ...
        ranksum_stat_importance_filt_ref, ranksum_stat_importance_ker_ref, ranksum_stat_importance_tran_ref, ...
        ranksum_stat_importance_invar_ref, ranksum_stat_importance_feature_ref, ranksum_stat_importance_all_ref] = ...
            calcStatImportance(ranksum_h_ref, func_name, bin_theta, rho, filter_name, kernel_style, nonlin_trans, n_max, l_max, features);
    
    
                [t_stat_importance_func_ref, t_stat_importance_theta_ref, t_stat_importance_rho_ref, ...
        t_stat_importance_filt_ref, t_stat_importance_ker_ref, t_stat_importance_tran_ref, ...
        t_stat_importance_invar_ref, t_stat_importance_feature_ref, t_stat_importance_all_ref] = ...
            calcStatImportance(t_h_ref, func_name, bin_theta, rho, filter_name, kernel_style, nonlin_trans, n_max, l_max, features);

        select_num_hyps_false_corr = ranksum_num_hyps_false_corr;
        stat_importance_all_rank = ranksum_stat_importance_all_ref;
        stat_importance_all_t = t_stat_importance_all_ref;
    else
        select_num_hyps_false_corr = ranksum_num_hyps_false;
    end

    select_test_ind = ranksum_ind_nans;
    if(select_num_hyps_false_corr == 0)
        disp('Error: No statistically significant features found! Proceding with features without FDR refinement!');
        pos_imgs_feats_signif = NaN;
        neg_imgs_feats_signif = NaN;
    else

        pos_imgs_feats_signif = getSignifFeatures(pos_imgs_feats, select_test_ind, select_num_hyps_false_corr, ...
    func_name, bin_theta, rho, filter_name, kernel_style, nonlin_trans, n_max, l_max, features);

        neg_imgs_feats_signif = getSignifFeatures(neg_imgs_feats, select_test_ind, select_num_hyps_false_corr, ...
    func_name, bin_theta, rho, filter_name, kernel_style, nonlin_trans, n_max, l_max, features);
    end
end

%% Sort Test Results
% sort statistical test resutls

% inputs: test_p - test p-values
%       : test_h - hypotheses acceptance binary
% outputs: test_p_sort - sorted test p-values without NaNs
%        : test_p_sort_nans - sorted test p-values with NaNs included
%        : test_h_sort - sorted hypotheses acceptance binary without NaNs
%        : test_h_sort_nans - sorted hypotheses acceptance binary with NaNs
%        : ind - sorting without NaNs indices
%        : ind_nans - sorting with NaNs indices

function [test_p_sort, test_p_sort_nans, test_h_sort, test_h_sort_nans, ind, ind_nans] = sortTestResults(test_p, test_h)
    test_p = reshape(test_p, [], 1);
    test_h = reshape(test_h, [], 1);
    test_p_nans = test_p; 
    test_h_nans = test_h;
    test_p(isnan(test_h)) = [];
    test_h(isnan(test_h)) = [];

    [test_p_sort_nans, ind_nans] = sort(test_p_nans);
    [test_p_sort, ind] = sort(test_p);
    test_h_sort_nans = test_h_nans(ind_nans);
    test_h_sort = test_h(ind);
end

%% Find P-Value Threshold Index
% find the index of first p-value that does not fulfill FDR refinement

% inputs: num_hyps - total number of hypotheses tested
%       : alpha - original p-value
%       : test_p_sort - sorted test p-values without NaNs
% outputs: pval_threshold_ind - index of first p-value that does not
%                               fulfill FDR refinement

function [pval_threshold_ind] = findPValThresholdInd(num_hyps, alpha, test_p_sort)
    harm_num = double(log(num_hyps) + eulergamma + 1/(2*num_hyps));
    pval_threshold = (((1:num_hyps) ./ (num_hyps * harm_num)) * alpha)';
    pval_diffs = test_p_sort - pval_threshold;
    pval_threshold_ind = find(pval_diffs > 0, 1, 'first');
end

%% Refine FDR
% refine hypotheses acceptance binary array

% inputs: test_pval_threshold_ind - index of first p-value that does not
%                                   fulfill FDR refinement 
%       : test_h_sort - hypotheses acceptance binary array
% outputs: test_h_sort - refined hypotheses acceptance binary array

function [test_h_sort] = refineFDR(test_pval_threshold_ind, test_h_sort)
    if(numel(test_pval_threshold_ind) > 0 && test_pval_threshold_ind > 1)
        test_h_sort(1:test_pval_threshold_ind-1) = 1;
        test_h_sort(test_pval_threshold_ind:end) = 0;
    elseif(numel(test_pval_threshold_ind) == 0)
        test_h_sort(1:end) = 1;
    elseif(test_pval_threshold_ind == 1)
        test_h_sort(1:end) = 0;
    end
end

%% Get Significant Features
% get array of significant features from array of all image features

% inputs: img_feats - features of either all positive or negative images
%       : test_ind - sorting indices according to chosen stat. test
%       : num_hyps_false_corr - number of statistically significant
%                               features
%       : func_name - function used for invariant calculation
%       : bin_theta - binary mask threshold
%       : rho - function radius
%       : filter_name - names of used filters
%       : kernel_style - names of used filtering kernels
%       : nonlin_trans - names of used nonlinear transformations
%       : n_max - invariant max. n 
%       : l_max - invariant max. l
%       : features - names of calculated features
% outputs: img_features_signif - significant feature values for all images

function [img_features_signif] = getSignifFeatures(img_feats, test_ind, num_hyps_false_corr, ...
    func_name, bin_theta, rho, filter_name, kernel_style, nonlin_trans, n_max, l_max, features)
    num_pics = size(img_feats, 1);  
    img_features_signif = zeros(num_pics, num_hyps_false_corr);
    for i = 1:num_pics
        cur_features = reshape(img_feats(i, :, :, :, :, :, :, :, :), [], 1);
        cur_features = cur_features(test_ind);
        cur_features = cur_features(1:num_hyps_false_corr);
        img_features_signif(i, :) = cur_features;
    end

    var_names = createVarNames(func_name, bin_theta, rho, filter_name, kernel_style, nonlin_trans, n_max, l_max, features);
    var_names = reshape(var_names, [], 1);
    var_names = var_names(test_ind);
    var_names = var_names(1:num_hyps_false_corr);
    img_features_signif = array2table(img_features_signif, 'VariableNames', var_names');
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