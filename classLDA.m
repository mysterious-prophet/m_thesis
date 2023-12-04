%% Classify Using LDA
% classify input using Linera Discriminant Analysis

% inputs: x - sample to be classified
%       : res_train - results of LDA training
% outputs: y - classified sample

function y = classLDA(x, res_train)
    mu = res_train{1};
    sigma_inv = res_train{2};
    sigma_log_det = res_train{3};
    num_classes = size(mu, 1);
    lda_func = zeros(num_classes, 1);

    for i = 1:num_classes
        class_mu = mu(i, :);
        lda_func(i) = sigma_log_det + (x - class_mu')' * sigma_inv * (x - class_mu');
    end

    [~, ind_class] = min(lda_func);
    if(numel(ind_class) == 1)
        y = ind_class - 1;
    else
        y = 0;
    end
end