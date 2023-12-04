%% Train KNN
% train K Nearest Neighbors classifier

% inputs: X - inputs for training
%       : y_star - class labels
%       : train_params - classifier training parameters
% outputs: res_train - parameters for KNN classification

function res_train = trainKNN(X, y_star, train_params)
    knn = train_params(1);
    p = train_params(2);
    res_train = {X, y_star, knn, p};
end