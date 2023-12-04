%% Classify ANN
% classify input using Matlab neural network model

% inputs: x - input to be classified
%       : res_train - trained ANN matlab model
% outputs: y - classified input

function y = classANN(x, res_train)
    ann_model = res_train{1};
    x = x';
    y = predict(ann_model, x);
end