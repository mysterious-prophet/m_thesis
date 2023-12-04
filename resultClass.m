%% Result Class
% class defining results/outputs such as image features, training datasets,
% classification results, etc. 

classdef resultClass
    properties
        % Calculated image features
        pos_imgs_features = NaN;
        neg_imgs_features = NaN;

        % Significant image features after statistical testing, best
        % configuration acc. to ranksum and t-test, number of statistically
        % significant features after FDR refinement
        pos_imgs_features_signif = NaN; 
        neg_imgs_features_signif = NaN;
        ranksum_stat_importance_all = NaN;
        t_stat_importance_all = NaN;
        select_num_hyps_false_corr = NaN;

        % Classifier inputs
        X_train = NaN;
        X_train_whit = NaN;

        % Tables of classification statistics
        knn_opt = NaN;
        class_stats_leave1 = NaN;
        class_stats_stratkfold = NaN;
        class_stats_leave1_triv = NaN;
        signif_feats_class_stats_leave1 = NaN;
        class_stats_stratkfold_triv = NaN;
        class_stats_leave1_whit = NaN;
        class_stats_stratkfold_whit = NaN;

        % Accuracy figures for best classifiers
        class_acc_leave1_lda = [];
        class_acc_leave1_qda = [];
        class_acc_leave1_knn = [];
        class_acc_leave1_lda_whit = [];
        class_acc_leave1_qda_whit = [];
        class_acc_leave1_knn_whit = [];
        class_acc_leave1_svm_whit = [];
        class_acc_leave1_sigmoid_whit = [];
        class_acc_leave1_tanh_whit = [];
    end
end