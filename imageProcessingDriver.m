%% Image processing driver
% loads input image, adds noise to input image, processes noised image with
% selected filter, measures time and quality of processing in signal to
% noise ratio and saves results as table and figure

% frequency - lowpass - ideal, circular Bessel, polynomial Bessel, Gauss,
%                     - Butterworth, alpha-stable 
%           - highpass - see lowpass, laplacian

% inputs: X - image to be processed
%       : frame_style - 0, 1, 2, padding style
%       : kernel_style - string,  type of kernel used
%       : filter_name - string, name of filter used
%       : nonlin_trans - string, type of nonlinear transformation
%       : varargin - additional filter, nonlin_trans parameters
% outputs: Y - processed image

function Y = imageProcessingDriver(X, frame_style, kernel_style, filter_name, nonlin_trans, varargin)
    % frequency domain low-pass and high-pass filtering
    if(strcmp(filter_name, 'fftLP') || strcmp(filter_name, 'fftHP'))
        K = calcFilterKernel(X, filter_name, kernel_style, varargin);
        X_tran = calcNonLinTran(X, nonlin_trans, 'dir', varargin);
        Y_tran = filterFunctionFFT(X_tran, K, frame_style);
        Y = calcNonLinTran(Y_tran, nonlin_trans, 'rev', varargin);

    % frequency domain laplacian
    elseif(strcmp(filter_name, 'fftLap'))
        K = calcFilterKernel(X, filter_name, filter_name, varargin);
        X_tran = calcNonLinTran(X, nonlin_trans, varargin);
        Y_tran = filterFunctionFFT(X_tran, K, frame_style);
        Y = revNonLinTran(Y_tran, nonlin_trans, varargin);
        Y = X - Y;

    % frequency domain highboost filtering
    elseif(strcmp(filter_name, 'fftHB')) 
        K = calcFilterKernel(X, 'fftLP', kernel_style, varargin);
        X_tran = calcNonLinTran(X, nonlin_trans, varargin);
        Y_tran = filterFunctionFFT(X_tran, K, frame_style);
        Y = revNonLinTran(Y_tran, nonlin_trans, varargin);

        if(numel(varargin) < 1)
            k = 1;
        else
            k = varargin{end};
        end
        Y = X + k*(X - Y);
    end
end