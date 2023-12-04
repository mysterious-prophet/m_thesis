%% Save Results
% save properties of instance of results class into results/ folder

% inputs: configuration - instance of configuration class
%       : results -  instance of result class


function [] = saveResults(configuration, results)
    if(~isnan(results.pos_imgs_features(1, 1)))
        filename = getFilename(configuration, "AD_Images_Feats._");
        writeData(results.pos_imgs_features, "Image Features", filename, 0);
    end

    if(~isnan(results.neg_imgs_features(1, 1)))
        filename = getFilename(configuration, "CN_Images_Feats._");
        writeData(results.neg_imgs_features, "Image Features", filename, 0);
    end

    if(istable(results.pos_imgs_features_signif(1, 1)))
        filename = getFilename(configuration, "AD_Images_Signif._Feats._");
        writeData(results.pos_imgs_features_signif, "Statistical Significance", filename, 2);
    end

    if(istable(results.neg_imgs_features_signif(1, 1)))
        filename = getFilename(configuration, "CN_Images_Signif._Feats._");
        writeData(results.neg_imgs_features_signif, "Statistical Significance", filename, 2);
    end

    if(istable(results.ranksum_stat_importance_all))
        filename = getFilename(configuration, "Ranksum_Stat._Importance_");
        writeData(results.ranksum_stat_importance_all, "Statistical Significance", filename, 2);
    end

    if(istable(results.t_stat_importance_all))
        filename = getFilename(configuration, "T_Stat._Importance_");
        writeData(results.t_stat_importance_all, "Statistical Significance", filename, 2);
    end

    if(~isnan(results.select_num_hyps_false_corr(1, 1)))
        filename = getFilename(configuration, "Num._Signif._Hyps._");
        writeData(results.select_num_hyps_false_corr, "Statistical Significance", filename, 1);
    end

    if(~isnan(results.X_train(1, 1)) && (configuration.whiten_data))
        filename = getFilename(configuration, "Data_train_");
        writeData(results.X_train, "Data Whitening", filename, 0);
    end

    if(~isnan(results.X_train_whit(1, 1)) && (configuration.whiten_data))
        filename = getFilename(configuration, "Data_Train_Whit._");
        writeData(results.X_train_whit, "Data Whitening", filename, 0);
    end

    if(~isnan(results.knn_opt(1, 1)))
        filename = getFilename(configuration, "Knn_Opt._");
        writeData(results.knn_opt, "Classification", filename, 1);
    end

    if(istable(results.class_stats_leave1))
        filename = getFilename(configuration, "Class._Stats._Leave1_");
        writeData(results.class_stats_leave1, "Classification", filename, 2);
    end

    if(istable(results.class_stats_stratkfold))
        stratkfold_string = strcat("Class._Strat-", string(configuration.k_fold), "-Fold_");
        filename = getFilename(configuration, stratkfold_string);
        writeData(results.class_stats_stratkfold, "Classification", filename, 2);
    end

    
    if(istable(results.class_stats_leave1_triv))
        filename = getFilename(configuration, "Class._Stats._Leave1_Triv._");
        writeData(results.class_stats_leave1_triv, "Classification", filename, 2);
    end

    if(istable(results.signif_feats_class_stats_leave1))
        filename = getFilename(configuration, "Signif._Feats._Class._Stats._Leave1_");
        writeData(results.signif_feats_class_stats_leave1, "Classification", filename, 2);
    end

    if(istable(results.class_stats_stratkfold_triv))
        stratkfold_string = strcat("Class._Strat-", string(configuration.k_fold), "-Fold_Triv._");
        filename = getFilename(configuration, stratkfold_string);
        writeData(results.class_stats_stratkfold_triv, "Classification", filename, 2);
    end

    if(istable(results.class_stats_leave1_whit))
        filename = getFilename(configuration, "Class._Stats._Leave1_Whit._");
        writeData(results.class_stats_leave1_whit, "Classification", filename, 2);
    end

    if(istable(results.class_stats_stratkfold_whit))
        stratkfold_string = strcat("Class._Strat-", string(configuration.k_fold), "-Fold_Whit._");
        filename = getFilename(configuration, stratkfold_string);
        writeData(results.class_stats_stratkfold_whit, "Classification", filename, 2);
    end

    if(~isempty(results.class_acc_leave1_lda))
        filename = getFilename(configuration, "Class._Acc._LDA_");
        writeData(results.class_acc_leave1_lda, "Classification", filename, 3);
    end

    if(~isempty(results.class_acc_leave1_qda))
        filename = getFilename(configuration, "Class._Acc._QDA_");
        writeData(results.class_acc_leave1_qda, "Classification", filename, 3);
    end

    if(~isempty(results.class_acc_leave1_knn))
        filename = getFilename(configuration, "Class._Acc._KNN_");
        writeData(results.class_acc_leave1_knn, "Classification", filename, 3);
    end

    if(~isempty(results.class_acc_leave1_lda_whit))
        filename = getFilename(configuration, "Class._Acc._LDA_Whit._");
        writeData(results.class_acc_leave1_lda_whit, "Classification", filename, 3);
    end

    if(~isempty(results.class_acc_leave1_qda_whit))
        filename = getFilename(configuration, "Class._Acc._QDA_Whit._");
        writeData(results.class_acc_leave1_qda_whit, "Classification", filename, 3);
    end

    if(~isempty(results.class_acc_leave1_knn_whit))
        filename = getFilename(configuration, "Class._Acc._KNN_Whit._");
        writeData(results.class_acc_leave1_knn_whit, "Classification", filename, 3);
    end

    if(~isempty(results.class_acc_leave1_svm_whit))
        filename = getFilename(configuration, "Class._Acc._SVM_Whit._");
        writeData(results.class_acc_leave1_svm_whit, "Classification", filename, 3);
    end

    if(~isempty(results.class_acc_leave1_sigmoid_whit))
        filename = getFilename(configuration, "Class._Acc._Sigmoid_Whit._");
        writeData(results.class_acc_leave1_sigmoid_whit, "Classification", filename, 3);
    end 

    if(~isempty(results.class_acc_leave1_tanh_whit))
        filename = getFilename(configuration, "Class._Acc._Tanh_Whit._");
        writeData(results.class_acc_leave1_tanh_whit, "Classification", filename, 3);
    end 
end


%% Get Filename
% get filename for file saving

% inputs: configuration - instance of configuration class
%       : result_type -  type of result (features, classification, etc.)
% outputs: filename - filename, with which the property will be saved

function filename = getFilename(configuration, result_type)
    num_pos_imgs = string(configuration.num_pos_imgs);
    num_neg_imgs = string(configuration.num_neg_imgs);
    bin_theta = getArrayString(configuration.bin_theta);
    func_name = getArrayString(configuration.func_name);
    l_max = string(configuration.l_max);
    n_max = string(configuration.n_max);
    rho = getArrayString(configuration.rho);
    kernel_style = getArrayString(configuration.kernel_style);
    filter_name = getArrayString(configuration.filter_name);

    filename = strcat(result_type, "Num._AD=", num_pos_imgs, "_Num._CN=", num_neg_imgs, "_Theta=", bin_theta, ...
            "_Func(s).=", func_name, "_nMax=", n_max, "_lMax=", l_max, "_Rho=", rho, ...
            "_Filter(s)=", filter_name ,"_Kernel(s)=", kernel_style);
end

%% Write Data
% write data into a file

% inputs: data - data array/table/figure
%       : folder_name - folder, into which the result will be saved
%       : filename -  filename of file to be saved
%       : mode - saving either array/table/figure + png

function [] = writeData(data, folder_name, filename, mode)
    folder_name = "results/" + folder_name;
    if(~exist(folder_name, 'dir'))
        mkdir(folder_name);
    end

    if(mode == 0)
        filename = folder_name + "/" + filename + ".mat";
        save(filename, 'data');
    elseif(mode == 1)
        filename = folder_name + "/" + filename + ".csv";
        save(filename, 'data');
        writematrix(data, filename);
    elseif(mode == 2)
        filename = folder_name + "/" + filename + ".csv";
        writetable(data, filename, 'WriteRowNames', true);
    elseif(mode == 3)
        filename_fig = folder_name + "/" + filename + ".fig";
        filename_png = folder_name + "/" + filename + ".png";
        fig = figure(data);
        fullfig(fig);
        data.Visible = 'off';
        saveas(fig, filename_fig);
        export_fig(fig, filename_png, '-p0.02');
        close(data);
    end
end