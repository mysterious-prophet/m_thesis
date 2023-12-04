%% Calculate Invariant Function
% calculates invariant function 

% inputs: func_name - name of function for invariant calculations
%       : l, m, n - spherical harmonic function parameters
%       : rho - function radius
%       : M, N, P - input image size

function invar_func = calcInvarFunc(func_name, l, m, n, rho, M, N, P)
    % calculate omega 3D coordinates 
    [k, mu, psi] = calcFreqCoords(M, N, P);

    if(strcmp(func_name, 'zerPoly'))
        invar_func = (-1)^(n) / (1i^(l) * (2*pi)^(l + 1/2)) * ...
            calcSpherHarmFunc(l, m, mu ./(rho * k), psi ./(rho * k)) .* (besselj(2*n + l + 3/2, -rho * k) ./ (-rho * k));
    end

    invar_func(isnan(invar_func)) = 0;
    invar_func = conj(invar_func);
end