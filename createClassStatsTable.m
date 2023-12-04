%% Create Classifier Statistics Table
% creates classifier statistics table from array

% inputs: class_stats_arr - sorted array of classifier statistics
%       : var_names - sorted array of classifier names
% outputs: class_stats_table - table of classifier statistics

function class_stats_table = createClassStatsTable(class_stats_arr, var_names)
    class_stats_table = array2table(class_stats_arr, 'VariableNames', var_names, 'RowNames', ...
        {'TPR', 'TNR', 'FPR', 'FNR', 'Precision', 'Neg. pred. val.', 'False disc. rate', ...
        'False om. rate', 'Pos. likel. ratio', 'Neg. likel. ratio', 'Preval. thres.', ...
        'Threat score', 'Accuracy', 'Bal. accuracy', 'F1 score', 'Matthews corr. coeff.', ...
        'Fowlkes-Mallows index', 'Diagnostic odds ratio'});
end