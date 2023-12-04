%% Train LDA
% train Linear Discriminant Analysis

% inputs: X - inputs for training
%       : y_star - class labels
%       : lambda - LDA regularization
% outputs: res_train - parameters for LDA classification

function res_train = trainLDA(X, y_star, lambda)
    [num_imgs, num_features] = size(X);
    num_classes = size(unique(y_star), 1);
    mu = zeros(num_classes, num_features);
    sigma = zeros(num_features, num_features);

    for i = 1:num_classes
        ind_class = find(y_star == (i-1));
        num_found = numel(ind_class);
        if(num_found == 0)
            mu(i, :) = inf;
        else
            X_found = X(ind_class, :);
            mu(i, :) = mean(X_found, 1);
            sigma = sigma + cov(X_found)*num_found;
        end
    end
    sigma = sigma / num_imgs + lambda*eye(num_features, num_features);
    sigma_inv = pinv(sigma);
    sigma_log_det = log(det(sigma));
    res_train = {mu, sigma_inv, sigma_log_det};
end