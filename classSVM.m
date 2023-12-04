%% Classify SVM
% classify input using matlab Support Vector Machine model

% inputs: x - input to be classified
%       : res_train - trained SVM matlab model
% outputs: y - classified input

function y = classSVM(x, res_train)
    svm_model = res_train{1};
    x = x';
    y = predict(svm_model, x);
end