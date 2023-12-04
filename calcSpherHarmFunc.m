%% Calculate Spherical Harmonic Function
% calculates spherical harmonic function for input l, m, phi, theta using
% Legendre polynomials

% inputs: l, m - spherical harmonic parameters
%       : phi, theta - spherical harmonic coordinates
% outputs: Y_lm - spherical harmonic function 

function Y_lm = calcSpherHarmFunc(l, m, phi, theta)
    P_lm = legendre(l, cos(theta));
    if(m < 0)
        % https://en.wikipedia.org/wiki/Associated_Legendre_polynomials
        % section Definition for non-negative integer parameters ℓ and m
        P_lm = (-1)^m * factorial(l - m)/factorial(l + m) .* P_lm; 
    end
    if(l ~= 0)
        P_lm = squeeze(P_lm(abs(m)+1, :, :, :));
    end

    Y_lm = sqrt(((2*l + 1) * factorial(l - m)) / (4*pi * factorial(l + m))) .* P_lm .* exp(1i*m*phi);
end