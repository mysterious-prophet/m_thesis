%% Data Whitening Driver
% transform and whiten data using Principal Component Analysis for
% classification

% inputs: pos_imgs_features - features of images with positive trait
%       : neg_imgs_features - features of images without positive trait
%       : func_name - fucntion used for invariant calculation
%       : bin_theta - binary mask threshold
%       : rho - function radius
%       : filter_name - names of used filters
%       : kernel_style - names of used kernels
%       : nonlin_trans - names of used nonlinear transformations
%       : n_max - max. invariant n
%       : l_max - max. invariant l
%       : features - names of calculated features
%       : num_whitening features - number of features after datawhitening
% outputs: X_train - classifier input data, transformed, without PCA
%        : X_train_whit - classifier input data, transformed, with PCA

function [X_train, X_train_whit] = dataWhiteningDriver(pos_imgs_features, neg_imgs_features, ...
    func_name, bin_theta, rho, filter_name, kernel_style, nonlin_trans, ...
    n_max, l_max, features, num_whit_features)

     [num_pos_imgs, num_neg_imgs, num_funcs, num_thetas, num_rhos, num_filts, ...
    num_kers, num_trans, num_invars, num_features, num_num_whit_features] = ...
        getNums(pos_imgs_features, neg_imgs_features, func_name, bin_theta, rho, ...
        filter_name, kernel_style, nonlin_trans, n_max, l_max, features, num_whit_features);

    num_features_onedim = num_funcs * num_thetas * num_rhos * ...
        ((num_filts - 1) * (num_kers - 1) * num_trans + 1) * num_invars * num_features;

    pos_imgs_features_onedim = reshapeImageFeatures(pos_imgs_features, num_features_onedim);
    neg_imgs_features_onedim = reshapeImageFeatures(neg_imgs_features, num_features_onedim);
    X_train = [pos_imgs_features_onedim; neg_imgs_features_onedim];

    X_train_whit = zeros(num_num_whit_features, num_pos_imgs + num_neg_imgs, max(num_whit_features));
    for i = 1:num_num_whit_features
        [~, ~, ~, X_train_whit(i, :, 1:num_whit_features(i))] = whitenDataPCA(X_train, num_whit_features(i));
    end
end

%% Reshape Image Features
% reshape images' features from 9D arrays to 2D arrays

% inputs: imgs_features - images' features
%       : num_features_onedim - number of features if transformed into
%                               vector for each image
% outputs: imgs_features_onedim - array of features where each row is a vector
%                                 of all features for one image

function imgs_features_onedim = reshapeImageFeatures(imgs_features, num_features_onedim)
    num_imgs = size(imgs_features, 1);
    imgs_features_onedim = zeros(num_imgs, num_features_onedim);
    for i = 1:num_imgs
        imgs_features_temp = reshape(imgs_features(i, :, :, :, :, :, :, :, :), 1, []);
        img_features_temp = imgs_features_temp(~isnan(imgs_features_temp));
        imgs_features_onedim(i, :) = img_features_temp;
    end
end


function[num_pos_imgs, num_neg_imgs, num_funcs, num_thetas, num_rhos, num_filts, ...
    num_kers, num_trans, num_invars, num_features, num_num_whit_features] = ...
        getNums(pos_imgs_features, neg_imgs_features, func_name, bin_theta, rho, ...
        filter_name, kernel_style, nonlin_trans, n_max, l_max, features, num_whit_features)
    
    num_pos_imgs = size(pos_imgs_features, 1);
    num_neg_imgs = size(neg_imgs_features, 1);
    num_funcs = size(func_name, 2);
    num_thetas = size(bin_theta, 2);
    num_rhos = size(rho, 2);
    num_filts = size(filter_name, 2);
    num_kers = size(kernel_style, 2);
    num_trans = size(nonlin_trans, 2);
    num_invars = (n_max + 1) * (l_max + 1);
    num_features = size(features, 2);
    num_num_whit_features = size(num_whit_features, 2);
end

