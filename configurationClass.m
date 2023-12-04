%% Configuration Class
% class defining input configuration - image folders, filtering functions,
% classifiers, etc. 
% for numbered configurations see lines 142–478

classdef configurationClass
    properties (Access = public)
        % AD, CN images" datastore (if calculating features), filenames (if
        % only classifying) and number of images to work with
        pos_img_datastore;
        neg_img_datastore;
        num_pos_imgs;
        num_neg_imgs;

        % Binary mask threshold
        bin_theta;
       
        % SPECT normalization
        spect_norm;
         
        % Invariant calculation setup
        func_name;
        calc_image_features;
        l_max;
        n_max;
        rho;
        frame_style;
    
        % Invariant filtering - LP, HP - and filters - Gauss, Alpha-stable,
        % Butterworth
        filter_name;
        kernel_style;
        nonlin_trans;

        
    
        % Invariant statistical features
        features;
    
        % Statistical tests", p-value threshold, FDR refinement
        test_stat_importance;
        alpha;
        fdr_refine;
        
    
        % PCA setup
        whiten_data;
        num_whit_features;
        
    
        % Image classifiers for classification without data whitening
        % Filen with whitened data if only classifying
        X_train_filename;
        X_train_whit_filename;

        class_images;
        classifier_train_name;
        classifier_class_name;
        train_params;
    
        % Image classifiers for classification with data whitening
        classifier_train_name_whit;
        classifier_class_name_whit;
        train_params_whit;

        % Image classifiers for classification while using one-dimensional
        % data
        classifier_train_name_triv;
        classifier_class_name_triv;
        train_params_triv;

        % k-fold value for cross-validation
        k_fold;       
    end
    
    properties (Access = private)
        train_params_LDA_QDA = {0.001, 0.0025, 0.005, 0.0075, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, ...
            0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.5, ...
            2, 2.5, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 35 ...
            40, 45, 50, 60, 70, 80, 90, 100};
        train_params_KNN =  {[2 2], [3 2], [4 2], [5 2], [6 2], [7 2], [8 2], ...
            [9 2], [10 2], [11 2], [12 2], [13 2], [14 2], [15 2], [16 2], ...
            [17 2], [18 2], [19 2], [20 2]};
        train_params_SVM = {[1 0 0.5], [1 0.1 0.25], [1 0.1 0.75], ...
                [2 0 0.5], [2 0.1 0.25], [2 0.1 0.75] ...
                [0 0.1 0.5 0.01], [0 0.1 0.5 0.02], [0 0.1 0.5 0.03], [0 0.1 0.5 0.04], ...
                [0 0.1 0.5 0.05], [0 0.1 0.5 0.075], [0 0.1 0.5 0.1], ...
                [0 0.1 0.5 0.2], [0 0.1 0.5 0.3], [0 0.1 0.5 0.4], [0 0.1 0.5 0.5], ...
                [0 0.1 0.5 0.75], [0 0.1 0.5 1], [0 0.1 0.5 1.5], [0 0.1 0.5 2], ...
                [0 0.1 0.5 3], [0 0.1 0.5 4], [0 0.1 0.5 5], [0 0.1 0.5 7.5], [0 0.1 0.5 10]};

        train_params_ANN = {{3, "sigmoid", 1e-3}, {4, "sigmoid", 1e-3}, {5, "sigmoid", 1e-3}, {7, "sigmoid", 1e-3}, {8, "sigmoid", 1e-3}, {9, "sigmoid", 1e-3}, {10, "sigmoid", 1e-3}, ...
            {12, "sigmoid", 1e-3}, {14, "sigmoid", 1e-3}, {16, "sigmoid", 1e-3}, {18, "sigmoid", 1e-3}, {20, "sigmoid", 1e-3}, ...
            {3, "tanh", 1e-3}, {4, "tanh", 1e-3}, {5, "tanh", 1e-3}, {7, "tanh", 1e-3}, {8, "tanh", 1e-3}, {9, "tanh", 1e-3}, {10, "tanh", 1e-3}, ...
            {12, "tanh", 1e-3}, {14, "tanh", 1e-3}, {16, "tanh", 1e-3}, {18, "tanh", 1e-3}, {20, "tanh", 1e-3}};
    end

    methods
        function num_whit_features = setWhitFeatures(conf)
            for j = 1:size(conf.num_whit_features, 2)
                if(conf.num_whit_features(j) > min(conf.num_pos_imgs, conf.num_neg_imgs))
                    conf.num_whit_features(j) =  min(conf.num_pos_imgs, conf.num_neg_imgs);
                end
            end
            num_whit_features = unique(conf.num_whit_features);
        end

        function train_params = setLearnParams(conf)
            train_params = {conf.train_params_LDA_QDA, conf.train_params_LDA_QDA, conf.train_params_KNN};
        end

        function train_params_whit = setLearnParamsWhit(conf)
            train_params_whit = {conf.train_params_LDA_QDA, conf.train_params_LDA_QDA, conf.train_params_KNN, ...
                conf.train_params_SVM, conf.train_params_ANN};
        end

        function train_params_triv = setLearnParamsTriv(~, knn_opt)
            train_params_triv = {[knn_opt 2]};
        end

        function conf = configurationClass(conf_num)
            switch(conf_num)
                case 11
                    conf.pos_img_datastore = imageDatastore("data\AD_BARTOS\","FileExtensions", ".img", "ReadFcn", @loadImage);
                    conf.neg_img_datastore = imageDatastore("data\NOS0\", "FileExtensions", ".img", "ReadFcn", @loadImage);
                    conf.num_pos_imgs = 50;
                    conf.num_neg_imgs = 50;
                
                    conf.bin_theta = 0.2; 
                
                    conf.calc_image_features = true;
                    conf.func_name = "zerPoly";
                    conf.l_max = 2;
                    conf.n_max = 2;
                    conf.rho = 7;
                
                    conf.filter_name = ["noFilt" "fftLP"];
                    conf.kernel_style = ["noKer" "buttKer"];
                    conf.nonlin_trans = "noTran";
                
                    conf.features = ["max", "min", "range", "mean", "med", "mad", "1Q", "3Q", "IQR", "var", "skew", "kurt"];
                
                    conf.test_stat_importance = true;
                
                    conf.whiten_data = true;
                    
                    conf.class_images = true;

                case 12
                    conf.pos_img_datastore = imageDatastore("data\AD_BARTOS\","FileExtensions", ".img", "ReadFcn", @loadImage);
                    conf.neg_img_datastore = imageDatastore("data\NOS0\", "FileExtensions", ".img", "ReadFcn", @loadImage);
                    conf.num_pos_imgs = 50;
                    conf.num_neg_imgs = 50;

                    conf.bin_theta = 0.2; 

                    conf.calc_image_features = true;
                    conf.func_name = "zerPoly";
                    conf.l_max = 2;
                    conf.n_max = 2;
                    conf.rho = 7;

                    conf.filter_name = ["noFilt" "fftLP"];
                    conf.kernel_style = ["noKer" "gaussKer" "alpStKer" "buttKer"];
                    conf.nonlin_trans = "noTran";

                    conf.features = ["max", "min", "range", "mean", "med", "mad", "1Q", "3Q", "IQR", "var", "skew", "kurt"];

                    conf.test_stat_importance = true;

                    conf.whiten_data = true;

                    conf.class_images = true;

                case 13
                    conf.pos_img_datastore = imageDatastore("data\AD_BARTOS\","FileExtensions", ".img", "ReadFcn", @loadImage);
                    conf.neg_img_datastore = imageDatastore("data\NOS0\", "FileExtensions", ".img", "ReadFcn", @loadImage);
                    conf.num_pos_imgs = 50;
                    conf.num_neg_imgs = 50;

                    conf.bin_theta = 0.2; 

                    conf.calc_image_features = true;
                    conf.func_name = "zerPoly";
                    conf.l_max = 2;
                    conf.n_max = 2;
                    conf.rho = 7;

                    conf.filter_name = ["noFilt" "fftLP" "fftHP"];
                    conf.kernel_style = ["noKer" "gaussKer" "alpStKer" "buttKer"];
                    conf.nonlin_trans = ["noTran" "logTran"];

                    conf.features = ["max", "min", "range", "mean", "med", "mad", "1Q", "3Q", "IQR", "var", "skew", "kurt"];

                    conf.test_stat_importance = true;

                    conf.whiten_data = true;

                    conf.class_images = true;

                case 14
                    conf.pos_img_datastore = imageDatastore("data\AD_BARTOS\","FileExtensions", ".img", "ReadFcn", @loadImage);
                    conf.neg_img_datastore = imageDatastore("data\NOS0\", "FileExtensions", ".img", "ReadFcn", @loadImage);
                    conf.num_pos_imgs = 50;
                    conf.num_neg_imgs = 50;

                    conf.bin_theta = [0.05 0.1 0.2 0.3]; 

                    conf.calc_image_features = true;
                    conf.func_name = "zerPoly";
                    conf.l_max = 2;
                    conf.n_max = 2;
                    conf.rho = [3 5 7.5 10];

                    conf.filter_name = ["noFilt" "fftLP" "fftHP"];
                    conf.kernel_style = ["noKer" "gaussKer" "alpStKer" "buttKer"];
                    conf.nonlin_trans = ["noTran" "logTran"];

                    conf.features = ["max", "min", "range", "mean", "med", "mad", "1Q", "3Q", "IQR", "var", "skew", "kurt"];

                    conf.test_stat_importance = true;

                    conf.whiten_data = true;

                    conf.class_images = true;

                case 15
                    conf.pos_img_datastore = imageDatastore("data\AD_BARTOS\","FileExtensions", ".img", "ReadFcn", @loadImage);
                    conf.neg_img_datastore = imageDatastore("data\NOS0\", "FileExtensions", ".img", "ReadFcn", @loadImage);
                    conf.num_pos_imgs = 50;
                    conf.num_neg_imgs = 50;
                
                    conf.bin_theta = 0.2; 
                
                    conf.calc_image_features = true;
                    conf.func_name = "zerPoly";
                    conf.l_max = 3;
                    conf.n_max = 3;
                    conf.rho = 7;
                
                    conf.filter_name = ["noFilt" "fftLP"];
                    conf.kernel_style = ["noKer" "buttKer"];
                    conf.nonlin_trans = "noTran";
                
                    conf.features = ["max", "min", "range", "mean", "med", "mad", "1Q", "3Q", "IQR", "var", "skew", "kurt"];
                
                    conf.test_stat_importance = true;
                
                    conf.whiten_data = true;
                    
                    conf.class_images = true;
            end

            % SPECT normalization
            conf.spect_norm = 1;

            conf.frame_style = 0;

            conf.alpha = 0.05;
            conf.fdr_refine = true;

            conf.num_whit_features = [2 3 4 5 6 7 8 9 10];

            conf.classifier_train_name = ["trainLDA" "trainQDA" "trainKNN"];
            conf.classifier_class_name = ["classLDA" "classQDA" "classKNN"];
            conf.train_params = {};
        
            conf.classifier_train_name_whit = ["trainLDA" "trainQDA" "trainKNN" "trainSVM" "trainANN"];
            conf.classifier_class_name_whit = ["classLDA" "classQDA" "classKNN" "classSVM" "classANN"];
            conf.train_params_whit = {};
    
            conf.classifier_train_name_triv = "trainKNN";
            conf.classifier_class_name_triv = "classKNN";
            conf.train_params_triv = {};
    
            conf.k_fold = 10;  

            conf.num_whit_features = conf.setWhitFeatures();
            conf.train_params = conf.setLearnParams();
            conf.train_params_whit = conf.setLearnParamsWhit();
        end
    end
end