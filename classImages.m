%% Classify Images
% train classifier and cross-validate using leave-one-out or
% stratifier-k-fold methods

% inputs: X_train - array of inputs
%       : y_star_train - array of input labels
%       : num_pos_imgs_train - number of images with positive trait in
%                             training dataset
%       : num_neg_imgs_train - number of images without positive trait in
%                             training dataset
%       : classifier_train_name - name of classifier training function
%       : classifier_class_name - name of classifier classification
%                                 function
%       : k_fold - k fold value for stratified-k-fold cross-validation
%       : train_params - classifier training parameters

function [class_stats_leave1, class_stats_stratkfold] = ...
    classImages(X_train, y_star_train, num_pos_imgs_train, num_neg_imgs_train, ...
    classifier_train_name, classifier_class_name, k_fold, train_params)

    num_imgs_train = num_pos_imgs_train + num_neg_imgs_train;
    Y_train = zeros(num_imgs_train, 1);
    num_steps = min(floor(num_pos_imgs_train / k_fold), floor(num_neg_imgs_train / k_fold));
    class_stats_stratkfold = zeros(19, num_steps);

    if(rem(k_fold, 2) == 0)
        Y_stratkfold = zeros(k_fold, num_steps);
        X_ad = X_train(1:num_pos_imgs_train, :);
        X_cn = X_train(num_pos_imgs_train + 1:end, :);
        y_star_ad = y_star_train(1:num_pos_imgs_train);
        y_star_cn = y_star_train(num_pos_imgs_train + 1:end);
        for i = 1:num_steps
            X_train_temp = [X_ad(1:i-k_fold/2+1, :); X_ad(i+k_fold/2:num_pos_imgs_train, :); X_cn(1:i-k_fold/2+1, :); X_cn(i+k_fold/2:num_neg_imgs_train, :)];
            y_star_train_temp = [y_star_ad(1:i-k_fold/2+1); y_star_ad(i+k_fold/2:num_pos_imgs_train); y_star_cn(1:i-k_fold/2+1); y_star_cn(i+k_fold/2:num_neg_imgs_train)]; 
            X_class = [X_ad(i:i+k_fold/2-1, :); X_cn(i:i+k_fold/2-1, :)];
            y_star_class = [y_star_ad(i:i+k_fold/2-1); y_star_cn(i:i+k_fold/2-1)];
            res_train_stratkfold = feval(classifier_train_name, X_train_temp, y_star_train_temp, train_params);
            for j = 1:k_fold
                Y_stratkfold(j, i) = feval(classifier_class_name, X_class(j, :)', res_train_stratkfold);
            end
            class_stats_stratkfold(:, i) = calcClassStats(Y_stratkfold(:, i), y_star_class, k_fold/2, k_fold/2);
        end
    end
    class_stats_stratkfold = mean(class_stats_stratkfold, 2);

    perm_ind_train = randperm(num_imgs_train)';
    X_train = X_train(perm_ind_train, :);
    y_star_train = y_star_train(perm_ind_train);
    for i = 1:num_imgs_train
        X_train_temp = [X_train(1:i-1, :); X_train(i+1:num_imgs_train, :)];
        y_star_train_temp = [y_star_train(1:i-1); y_star_train(i+1:num_imgs_train)];
        res_train_leave1 = feval(classifier_train_name, X_train_temp, y_star_train_temp, train_params);
        Y_train(i, 1) = feval(classifier_class_name, X_train(i, :)', res_train_leave1);
    end
    class_stats_leave1 = calcClassStats(Y_train, y_star_train, num_pos_imgs_train, num_neg_imgs_train);
end

%% Calculate Classification Statistics
% calculates classification statistics such as TPR, TNR, ACC., etc.

% inputs: Y - classifier classification labels vector
%       : y_star - true labels vector
%       : num_pos_imgs - number of images with positive trait
%       : num_neg_imgs - number of images without positive trait
% outputs: class_stats - classification statistics table

function class_stats = calcClassStats(Y, y_star, num_pos_imgs, num_neg_imgs)
    num_imgs = size(y_star, 1);
    tn = 0; tp = 0;
    fn = 0; fp = 0;
    for i = 1:num_imgs
        if(Y(i) == 0 && y_star(i) == 0)
            tn = tn + 1;
        elseif(Y(i) == 1 && y_star(i) == 1)
            tp = tp + 1;
        elseif(Y(i) == 0 && y_star(i) == 1)
            fn = fn + 1;
        elseif(Y(i) == 1 && y_star(i) == 0)
            fp = fp + 1;
        end
    end
    tpr = tp / num_pos_imgs;
    tnr = tn / num_neg_imgs;
    se_star = min(tpr, tnr);
    fpr = fp / num_neg_imgs;
    fnr = fn / num_pos_imgs;
    ppv = tp / (tp + fp);
    npv = tn / (tn + fn);
    fdr = fp / (fp + tp);
    faor = fn / (fn + tn);
    lr_plus = tpr / fpr;
    lr_minus = fnr / tnr;
    pt = sqrt(fpr) / (sqrt(tpr) + sqrt(fpr));
    ts = tp / (tp + fn + fp);
    acc = (tp + tn) / (tp + tn + fn + fp);
    ba = (tpr + tnr) / 2;
    F1 = 2*tp/(2*tp + fp + fn);
    mcc = (tp * tn - fp * fn) / (sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)));
    fm = sqrt(ppv * tpr);
    dor = lr_plus / lr_minus;

    class_stats = [tpr tnr se_star fpr fnr ppv npv fdr faor lr_plus lr_minus pt ts acc ba F1 mcc fm dor]';
end