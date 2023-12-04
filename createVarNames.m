%% Create Variable Names
% create variable names array for statistical testing

% inputs: func_name - invariant function
%       : bin_theta - binary mask threshold
%       : rho - function radius
%       : filter_name - names of used filters
%       : kernel_style - names of used kernels
%       : nonlinear_trans - names of used nonlinear transformations
%       : n_max - max. invariant n
%       : l_max - max. invariant l
%       : features - names of calculated features
% outputs: var_names - 8D array containing names of variables for
%                      statistical testing

function var_names = createVarNames(func_name, bin_theta, rho, filter_name, kernel_style, nonlin_trans, n_max, l_max, features)

    [num_funcs, num_thetas, num_rhos, num_filts, num_kers, num_trans, num_invars, num_features, ~] = ...
            getNums(func_name, bin_theta, rho, filter_name, kernel_style, nonlin_trans, n_max, l_max, features);

    var_names = strings(num_funcs, num_thetas, num_rhos, num_filts, num_kers, num_trans, num_invars, num_features); 
    for i = 1:num_funcs
        for j = 1:num_thetas
            for k = 1:num_rhos
                for l = 1:num_filts
                    for m = 1:num_kers
                        for n = 1:num_trans
                            for p = 1:num_invars
                                for r = 1:num_features
                                    invar_n = ceil(p/(n_max+1)) - 1;
                                    invar_l = p - (n_max+1)*invar_n - 1;
                                    var_names(i, j, k, l, m, n, p, r) = strcat(func_name(i), ...
                                        ', ', string(bin_theta(j)), ', ', string(rho(k)), ...
                                        ', ', filter_name(l), ', ', kernel_style(m), ...
                                        ', ', nonlin_trans(n), ', n=', string(invar_n), ...
                                        ', l=', string(invar_l), ', ', features(r));
                                end
                            end
                        end
                    end
                end
            end
        end
    end
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