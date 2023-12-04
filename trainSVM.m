%% Train SVM
% train Matlab Support Vector Machine

% inputs: X - inputs for training
%       : y_star - class labels
%       : train_params - classifier training parameters
% outputs: res_train - trained SVM model

function res_train = trainSVM(X, y_star, train_params)
    type = train_params(1);
    outlier_coeff = train_params(2);
    nu = train_params(3);
    if(size(train_params, 2) > 3)
        sigma = train_params(4);
    end
    if(type > 0)
        svm_model = fitcsvm(X, y_star,'KernelFunction','polynomial', 'KernelScale','auto', ...
            'PolynomialOrder', type, 'OutlierFraction', outlier_coeff, 'Nu', nu, 'Standardize', true);
    elseif(type == 0)
        svm_model = fitcsvm(X, y_star,'KernelFunction', 'rbf', 'KernelScale', 2*sigma^2, ...
             'OutlierFraction', outlier_coeff, 'Nu', nu, 'Standardize', true);
    end

    res_train = {svm_model};
end