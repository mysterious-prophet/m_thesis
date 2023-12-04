%% Classify Using KNN
% classify input using K Nearest Neighbors algorithm

% inputs: x - sample to be classifier
%       : res_train - results of KNN training
% outputs: y - classifier sample


function y = classKNN(x, res_train)
    X = res_train{1};
    y_star = res_train{2};
    knn = res_train{3};
    p = res_train{4};

    num_imgs = size(y_star, 1);
    num_classes = size(unique(y_star), 1);
    knn_func = zeros(num_imgs, 1);

    for i = 1:num_imgs
        knn_func(i) = norm(X(i, :) - x', p);
    end

    knn_func_sort = sort(knn_func);
    knn_func_star = knn_func_sort(knn);
    y_sel = y_star(knn_func <= knn_func_star);

    knn_hist = histcounts(y_sel, num_classes);
    [~, ind_class] = max(knn_hist);
    if(numel(ind_class) == 1)
        y = ind_class - 1;
    else
        y = 0;
    end
end