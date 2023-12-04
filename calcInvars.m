%% Calculate Invariants
% calculates image invariants by summing invariant coefficients

% inputs: input_image - normalized input image
%       : func_name - function to be used for invariant calculation
%       : frame_style - frame style to be used for invariant FFT
%                       calculation
%       : l_max - max. spherical harmonic l
%       : n_max - max. spherical harmonic n
%       : rho - function radius
% outputs: invars_nl - all image invariants from 0 to n and 0 to l 

function invars_nl = calcInvars(input_image, func_name, frame_style, l_max, n_max, rho)
    [M, N, P] = size(input_image);
    invars_nl = zeros((n_max + 1) * (l_max + 1), M, N, P);
    for n = 0:n_max
        for l = 0:l_max
            coeffs_nlm = zeros(2*l + 1, M, N, P);

            ft_func_m_zero = calcInvarFunc(func_name, l, 0, n, rho, 2*N, 2*M, 2*P);
            coeffs_nlm(l+1, :, :, :) = abs(filterFunctionFFT(input_image, ft_func_m_zero, frame_style)).^2;

            if(l > 0)
                for m = 1:l
                    ft_func_m_minus = calcInvarFunc(func_name, l, -m, n, rho, 2*N, 2*M, 2*P);
                    ft_func_m_plus = calcInvarFunc(func_name, l, m, n, rho, 2*N, 2*M, 2*P);
        
                    coeffs_nlm(m, :, :, :) = abs(filterFunctionFFT(input_image, ft_func_m_minus, frame_style)).^2;
                    coeffs_nlm(2*l + 2 - m, :, :, :) = abs(filterFunctionFFT(input_image, ft_func_m_plus, frame_style)).^2;
                end
            end
            invars_nl((n_max + 1)*n + l + 1, :, :, :) = reshape(sum(coeffs_nlm, 1), [M N P]);
        end
    end
end