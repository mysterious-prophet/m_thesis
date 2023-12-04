%% Calculate Binary Mask
% calculates binary mask for normalized image X based on threshold theta

% inputs: X - normalized input image
%       : theta - binary mask threshold
% outputs: bin_mask - binary mask for image X 

function bin_mask = calcBinMask(X, theta)
    X(X < theta) = 0;
    X(X >= theta) = 1;
    bin_mask = reshape(X, [], 1);
end