%% Train ANN
% train Matlab Neural Network

% inputs: X - inputs for training
%       : y_star - class labels
%       : train_params - classifier training parameters
% outputs: res_train - trained ANN model

function res_train = trainANN(X, y_star, train_params)
    layer_sizes = train_params{1};
    layer_acts = train_params{2};
    reg = train_params{3};

    ann_model = fitcnet(X, y_star, 'LayerSizes', layer_sizes, 'Activations', layer_acts, 'Lambda', reg, ...
        'LayerBiasesInitializer', 'ones', 'Standardize', true);

    res_train = {ann_model};
end