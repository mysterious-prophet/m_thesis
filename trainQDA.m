%% Train QDA
% train Quadratic Discriminant Analysis

% inputs: X - inputs for training
%       : y_star - class labels
%       : lambda - QDA regularization
% outputs: res_train - parameters for QDA classification

function res_train = trainQDA(X, y_star, lambda)
    [num_imgs, num_features] = size(X);
    num_classes = size(unique(y_star), 1);
    mu = zeros(num_classes, num_features);
    sigma_inv = zeros(num_features, num_features, num_classes);
    sigma_log_det = zeros(num_classes, 1);

    for i = 1:num_classes
        ind_class = find(y_star == (i-1));
        num_found = numel(ind_class);
        if(num_found == 0)
            mu(i, :) = inf;
            sigma_inv(:, :, i) = inf;
            sigma_log_det(i) = inf;
        else
            X_found = X(ind_class, :);
            mu(i, :) = mean(X_found, 1);
            sigma = cov(X_found)*num_found;
            sigma = sigma / num_imgs + lambda*eye(num_features, num_features);
            sigma_inv(:, :, i) = pinv(sigma);
            sigma_log_det(i) = log(det(sigma));
        end
    end
    res_train = {mu, sigma_inv, sigma_log_det};
end