%% Frequency domain filtering driver
% takes spatial input image X and frequency kernel K, transforms X into
% frequency domain and performs frequency domain filtering

% inputs: X - input image in spatial domain
%       : K - filtering kernel in frequency domain
%       : frame_style - padding style
% outputs: Y - spatial domain filtered image

function Y = filterFunctionFFT(X, K, frame_style)
    [m, n, p] = size(X);
    X = getPaddingFFT(X, frame_style);

    % transform input image X to frequency domain
    X = fftshift(fftn(X));    

    % perform filtering as element-wise matrix multiplication in frequency
    % domain
    Y = X .* K;

    % transform filtered image Y back to spatial domain
    Y = ifftn(ifftshift(Y));

    % select the part of the filtered image corresponding to the original
    % input image
    Y = Y(1:m, 1:n, 1:p);
end