%% Image nonlinear transformations
% performs power, log or Box-Cox transformation on spatial image

%% Image nonlinear transformations function
% inputs: X - input image
%       : tran_name - powTran, logTran, boxCoxTran
%                   - name of the selected transformation
%       : direction - direct or reverse transformation
%       : varargin - 1 or 2 additional parameters
%                  - c, gamma - parameters of power transformation
%                  - c, lambda - parameters of power transformation
%                  - lambda - parameter of Box-Cox transformation
% output: X - transformed image X

function X = calcNonLinTran(X, tran_name, direction, varargin)
    if(strcmp(direction, 'dir'))
        switch tran_name           
            case 'powTran'
                if(numel(varargin{1}) < 2)
                    c = 1;
                    gamma = 1;
                else
                    c = varargin{1}{end}(end-1);
                    if(c <= 0)
                        c = 1;
                    end
                    gamma = varargin{1}{end}(end);
                    if(gamma <= 0)
                        gamma = 1;
                    end
                end
                X = real(c * X.^gamma);
    
            case 'logTran'
                if(numel(varargin{1}) < 2)
                    c = 1;
                    lambda = 1;
                else
                    c = varargin{1}{end}(end-1);
                    if(c <= 0)
                        c = 1;
                    end
                    lambda = varargin{1}{end}(end);
                    if(lambda <= -min(X))
                        lambda = -min(X) + 1;
                    end
                end
                X = c * log(lambda + X);
    
            case 'boxCoxTran'
                if(numel(varargin{1}) < 1)
                    lambda = 1;
                else
                    lambda = varargin{1}{end};
                end
    
                if (lambda == 0)
                    X = log(max(X, 1));
                else
                    X = (max(X, 1).^lambda - 1)/lambda;
                end
    
        end
    elseif(strcmp(direction, 'rev'))
        switch tran_name    
            case 'powTran'
                if(numel(varargin{1}) < 2)
                    c = 1;
                    gamma = 1;
                else
                    c = varargin{1}{end}(end-1);
                    if(c <= 0)
                        c = 1;
                    end
                    gamma = varargin{1}{end}(end);
                    if(gamma <= 0)
                        gamma = 1;
                    end
                end
                X = (X / c) .^ (1 / gamma);
    
            case 'logTran'
                if(numel(varargin{1}) < 2)
                    c = 1;
                    lambda = 1;
                else
                    c = varargin{1}{end}(end-1);
                    if(c <= 0)
                        c = 1;
                    end
                    lambda = varargin{1}{end}(end);
                    if(lambda <= -min(X))
                        lambda = -min(X) + 1;
                    end
                end
                X = exp(X / c) - lambda;
    
            case 'boxCoxTran'
                if(numel(varargin{1}) < 1)
                    lambda = 1;
                else
                    lambda = varargin{1}{end};
                end
    
                X(X == 0) = 1;
                if (lambda == 0)
                    X(X ~= 0) = exp(X);
                else
                    X(X ~= 0) = (lambda * X + 1).^(1/lambda);
                end
    
        end
    end
end