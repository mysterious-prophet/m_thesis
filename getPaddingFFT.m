%% Frequency domain padding 
% pads image for filtering in frequency domain

% inputs: X - input image in spatial domain
%       : num_rows - number of rows to be added to image
%       : num_cols - number of columns to be added to image
%       : num_grids - 3D image height
%       : style -  0, 1, 2 - padding style
%                     - 0 - zero padding
%                     - 1 - copy padding
%                     - 2 - periodic padding
% outputs: X - padded image in spatial domain

function X = getPaddingFFT(X, style)
    [num_rows, num_cols, num_grids] = size(X);
    switch style
        % zero padding
        case 0 
            X = padarray(X, [num_rows num_cols num_grids], 0, 'post');        
        % copy padding
        case 1
            X = padarray(X, [num_rows num_cols num_grids], 'replicate', 'post');       
        % mirror padding
        case 2
            X = padarray(X, [num_rows num_cols num_grids], 'symmetric', 'post');
    end
end