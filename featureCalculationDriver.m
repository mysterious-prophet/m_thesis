%% Feature Calculation Driver
% driver for calculating features of images

% inputs: img_datastore - datastore containing folder with all images
%       : num_imgs - number of images to be taken from the folder
%       : bin_theta - binary mask threshold
%       : norm - image normalization type
%       : func_name - names of invariant calculation functions
%       : l_max - max. l for spherical harmonic functions
%       : n_max - max. n for spherical harmonic functions
%       : rho - radius of function
%       : frame_style - frame style used for invariant calculation and
%                       invariant filtering
%       : kernel_style - names of kernels used for invariant filtering
%       : filter_name -  names of filters used for invariant filtering
%       : nonlin_trans - names of nonlinear transformations used on
%                        filtered invariant
%       : features - names of features to be calculated on each invariant
% outputs: imgs_features - calculated features for all images

function imgs_features = featureCalculationDriver(img_datastore, ...
    num_imgs, bin_theta, norm, func_name, ...
    l_max, n_max, rho, frame_style, kernel_style, filter_name, nonlin_trans, features)
    
    imgs_features = [];
    if(num_imgs > 0)
        imgs_features = getImagesFeatures(img_datastore, num_imgs, bin_theta, ...
            norm, func_name, l_max, n_max, rho, frame_style, kernel_style,...
            filter_name, nonlin_trans, features);
    end
end


%% Get Images' Features
% get features of all input images

% inputs: img_datastore - datastore containing folder with images
%       : num_imgs - number of images to be selected from the folder
%       : bin_theta - binary mask threshold
%       : norm - image normalization type
%       : func_name - names of invariant calculation functions
%       : l_max - max. l for spherical harmonic functions
%       : n_max - max. n for spherical harmonic functions
%       : rho - radius of function
%       : frame_style - frame style used for invariant calculation and
%                       invariant filtering
%       : kernel_style - names of kernels used for invariant filtering
%       : filter_name -  names of filters used for invariant filtering
%       : nonlin_trans - names of nonlinear transformations used on
%                        filtered invariant
%       : features - names of features to be calculated on each invariant
% outputs: imgs_features - calculated features for all images

function[imgs_features] = getImagesFeatures(img_datastore, num_imgs, bin_theta, ...
    norm, func_name, l_max, n_max, rho, frame_style, kernel_style,...
    filter_name, nonlin_trans, features)

    [num_funcs, num_thetas, num_rhos, num_filts, num_kers, num_trans, num_invars, num_features] = ...
        getNums(func_name, bin_theta, rho, filter_name, kernel_style, nonlin_trans, n_max, l_max, features);

    imgs_features = NaN(num_imgs, num_funcs, num_thetas, num_rhos, num_filts, num_kers, num_trans, num_invars, num_features);
    rand_ind = randperm(size(img_datastore.Files, 1));
    if(num_imgs <= size(rand_ind, 2))
        rand_ind = rand_ind(1:num_imgs);
    end

    for i = 1:num_imgs
        for j = 1:num_funcs
            for k = 1:num_thetas
                [X, spect_bool] = loadImage(img_datastore.Files{rand_ind(i)});
                if(spect_bool)
                    X = normSpectImage(X, norm);
                end
                bin_mask = calcBinMask(X, bin_theta(k));
                for l = 1:num_rhos
                   invars = calcInvars(X, func_name(j), frame_style, l_max, n_max, rho(l));
                   for m = 1:num_filts
                        if(strcmp('noFilt', filter_name(m)))
                            imgs_features(i, j, k, l, m, 1, 1, :, :) = calcImageFeatures(invars, bin_mask, bin_theta(k), ...
                                func_name(j), l_max, n_max, rho(l), frame_style,...
                                kernel_style(1), filter_name(m), nonlin_trans(1), features);
                        
                        else
                            for n = 2:num_kers
                                for p = 1:num_trans
                                    imgs_features(i, j, k, l, m, n, p, :, :) = calcImageFeatures(invars, bin_mask, bin_theta(k), ...
                                        func_name(j), l_max, n_max, rho(l), frame_style,...
                                        kernel_style(n), filter_name(m), nonlin_trans(p), features);
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end


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
