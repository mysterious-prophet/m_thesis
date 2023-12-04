%% Calculate Invariant Feature
% calculate global feature of input invariant

% inputs: invar - input invariant
%       : feature_name - feature to be calculated
%       : varargin - mad, perc perameters
% outputs: invar_feature - calculated global feature of input invariant

function invar_feature = calcInvarFeature(invar, feature_name, varargin)
    invar_feature = NaN;
    if(strcmp(feature_name, 'min'))
        invar_feature = min(invar);
    elseif(strcmp(feature_name, 'max'))
        invar_feature = max(invar);
    elseif(strcmp(feature_name, 'range'))
        invar_feature = range(invar);
    elseif(strcmp(feature_name, 'mean'))
        invar_feature = mean(invar);
    elseif(strcmp(feature_name, 'med'))
        invar_feature = median(invar);
    elseif(strcmp(feature_name, 'mad'))
        if(nargin > 2)
            type = varargin{1};
        else
            type = 1;
        end
        invar_feature = mad(invar, type);
    elseif(strcmp(feature_name, 'perc'))
        if(nargin > 2)
            perc = varargin{1};
        else
            perc = 50;
        end
        invar_feature = prctile(invar, perc);
    elseif(strcmp(feature_name, 'var'))
        invar_feature = var(invar);
    elseif(strcmp(feature_name, 'IQR'))
        invar_feature = iqr(invar);
    elseif(strcmp(feature_name, 'skew'))
        invar_feature = skewness(invar);
    elseif(strcmp(feature_name, 'kurt'))
        invar_feature = kurtosis(invar);
    end
end