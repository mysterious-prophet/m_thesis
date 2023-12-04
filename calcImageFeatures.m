%% Calculate Image Features
% calculates features of input image invariants, optionally uses chosen
% filter on image invariants before feature calculation

% inputs: invars - invariants of input image
%       : bin_mask - binary mask corresponding to input image
%       : bin_theta - binary mask threshold
%       : func_name - name of function used for invariant calculation
%       : l_max - max. l of spherical harmonic function
%       : n_max - max. n of spherical harmonic function
%       : rho - radius of function used for invariant calculation
%       : frame_style - frame style used for invariant filtering
%       : kernel_style - kernel used for invariant filtering
%       : filter_name - filter used for invariant filtering
%       : nonlin_trans - nonlinear transformation used on filtered
%                       invariants
%       : features - features to be calculated
% outputs: image_features - calculated features of image invariants 


function image_features = calcImageFeatures(invars, bin_mask, bin_theta, ...
    func_name, l_max, n_max, rho, frame_style, kernel_style, ...
    filter_name, nonlin_trans, features)

    [~, ~, ~, ~, ~, ~, num_invars, num_features] = ...
        getNums(func_name, bin_theta, rho, filter_name, kernel_style, nonlin_trans, n_max, l_max, features);
   
    dims_invars = ndims(invars);
    if(dims_invars == 4)
        [~, M, N, P] = size(invars);
        image_features = zeros(num_invars, num_features);
        for i = 1:num_invars
            cur_invar = reshape(invars(i, :, :, :), [M N P]);            
            invar_max = max(max(max(cur_invar)));
            cur_invar = log(max(cur_invar / invar_max, 1e-4));

            if(~strcmp(filter_name, 'noFilt'))
                cur_invar = real(imageProcessingDriver(cur_invar, frame_style, kernel_style, filter_name, nonlin_trans));
            end

            cur_invar = reshape(cur_invar, [], 1);
            cur_invar = cur_invar(bin_mask == 1);
            for j = 1:num_features
                if(j == 7)
                    image_features(i, j) = calcInvarFeature(cur_invar, 'perc', 25);
                elseif(j == 8)
                    image_features(i, j) = calcInvarFeature(cur_invar, 'perc', 75);
                else
                    image_features(i, j) = calcInvarFeature(cur_invar, features(j));
                end
            end
        end
    end    
end


%% Get Numbers
% returns number of functions used for creating invariants etc.
% inputs: func_name - name of function used for invariant calculation
%       : bin_theta - binary mask threshold
%       : rho - radius of function used for invariant calculation
%       : filter_name -  filter used for invariant filtering
%       : kernel_style - kernel used for invariant filtering
%       : nonlin_trans - nonlinear transformation used on filtered
%                       invariants
%       : n_max - 
%       : l_max - 
%       : features - features to be calculated
% outputs: num_funcs - number of functions used for invariant calculation
%        : num_thetas - number of binary mask thresholds
%        : num_rhos - number of radii of function used for invariant calculation
%        : num_filts - number of filters used for invariant filtering
%        : num_kers -  number of kernels used for invariant filtering
%        : num_trans -  number of nonlinear transformation used on filtered
%                       invariants
%        : num_invars - number of invariants used for feature calculation
%        : num_features - number of features to be calculated

function[num_funcs, num_thetas, num_rhos, num_filts, num_kers, num_trans, num_invars, num_features] = ...
        getNums(func_name, bin_theta, rho, filter_name, kernel_style, nonlin_trans, n_max, l_max, features)

    num_funcs = size(func_name, 2);
    num_thetas = size(bin_theta, 2);
    num_rhos = size(rho, 2);
    num_filts = size(filter_name, 2);
    num_kers = size(kernel_style, 2);
    num_trans = size(nonlin_trans, 2);
    num_invars = (n_max + 1) * (l_max + 1);
    num_features = size(features, 2);
end