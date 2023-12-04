%% Whiten Data Using Principal Component Analysis
% inputs: X - data to be whitened
%       : num_dims - resulting number of dimenstions after PCA
% outputs: mean_X - averages of features
%        : lambda - eigenvalues
%        : W - weight vectors
%        : Y - whitened data

function [mean_X, lambda, W, Y] = whitenDataPCA(X, num_dims)
    [m, n] = size(X);
    if(n <= num_dims)
        return;
    end
    mean_X = mean(X, 1);
    
    X = bsxfun(@minus, X, mean_X);
    
    if(m < n)
        cov_mat = X * X' / sqrt(m - 1);
    else
        cov_mat = X' * X / (m - 1);
    end
    
    [W, L] = eig(cov_mat);
    lambda = diag(L);
    [lambda, ind] = sort(lambda, 'descend');
    ind = ind(1:num_dims);
    W = W(:, ind);
    
    if(m < n)
        W(:, 1:num_dims) = W(:, 1:num_dims) ./ max(lambda(1:num_dims), 1e-100)';
        W = X' * W;
    else
        W(:, 1:num_dims) = W(:, 1:num_dims) ./ (max(lambda(1:num_dims), 1e-100).^(1/2))';
    end
    
    lambda = cumsum(lambda);
    % principal component percentage
    lambda = 100*lambda(1:num_dims)' / lambda(end);
    mean_X = mean_X';
    Y = X * W; 
end